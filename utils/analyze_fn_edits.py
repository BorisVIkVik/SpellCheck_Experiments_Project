"""Analyze false-negative spellcheck edits with ERRANT categories (SAGE)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections import namedtuple
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from evaluate_prompt_errant import SPACY_MODEL, _ensure_sage_on_path, load_predictions

REQUIRED_COLUMNS = ("typed", "original", "prediction")
DEFAULT_INPUT = Path("prompt_eval_predictions.csv")
DEFAULT_OUTPUT = Path("fn_analysis.csv")
ProcessingArgs = namedtuple(
    "ProcessingArgs",
    ["dt", "ds", "single", "multi", "filt", "cse"],
    defaults=[False, False, False, False, [], True],
)


def _load_scorer(spacy_model: str = SPACY_MODEL):
    _ensure_sage_on_path()
    from sage.evaluation.scorer import Scorer

    return Scorer(load_errant=True, spacy_model=spacy_model)


def _processed_edits(scorer, source: str, target: str) -> dict[tuple[Any, ...], list[str]]:
    from errant.commands.compare_m2 import process_edits

    src_doc = scorer.errant.annotator.parse(source)
    tgt_doc = scorer.errant.annotator.parse(target)
    raw_edits = scorer.errant.annotate_errors(src_doc, tgt_doc)
    edit_dict = process_edits(raw_edits, ProcessingArgs())
    if not edit_dict:
        return {}
    coder_id = next(iter(edit_dict))
    edits = edit_dict[coder_id]
    return {
        key: categories
        for key, categories in edits.items()
        if key[0] != -1 and categories[0] != "noop"
    }


def _span_text(doc, start: int, end: int) -> str:
    if start < 0:
        return ""
    if start == end:
        if start < len(doc):
            return doc[start].text
        return ""
    return " ".join(token.text for token in doc[start:end])


def tag_fn_edit(
    *,
    fn_type: str,
    fn_gold_fix: str,
    typed: str,
    original: str,
    prediction: str,
    num_gold_edits: int,
    num_fn_in_sent: int,
) -> str:
    import re

    tags: list[str] = []

    if num_gold_edits >= 2:
        tags.append("multi_edit_sentence")
    if num_fn_in_sent >= 2:
        tags.append("multi_fn_in_sentence")

    if fn_type == "YO" or ("ё" in original and "ё" not in prediction):
        tags.append("yo")
    if fn_type == "CASE":
        tags.append("case")
    if fn_type == "PUNCT":
        tags.append("punct")

    if re.search(r"\w-\w", original) and not re.search(r"\w-\w", prediction):
        tags.append("hyphen_missing")
    if " -" in typed or "- " in typed:
        tags.append("hyphen_spacing")
    if len(original.split()) != len(prediction.split()):
        tags.append("token_count_change")

    if fn_type == "PUNCT" and fn_gold_fix and all(
        char in " ,.;:—–-…!?()" for char in fn_gold_fix
    ):
        tags.append("minor_punct")

    if re.search(r"[\u4e00-\u9fff]", prediction):
        tags.append("hallucination_non_ru")
    if len(prediction) > max(len(original), len(typed)) * 1.5:
        tags.append("overgeneration")

    return "|".join(tags) if tags else "other"


def _maybe_merge_device(
    fn_df: pd.DataFrame,
    device_repo: str | None,
    device_split: str,
) -> pd.DataFrame:
    if not device_repo or fn_df.empty:
        return fn_df

    from datasets import load_dataset

    device_df = load_dataset(device_repo, split=device_split).to_pandas()
    if "device_type" not in device_df.columns:
        raise ValueError(f"В {device_repo} ({device_split}) нет колонки device_type")

    merged = fn_df.merge(
        device_df[["typed", "original", "device_type"]].drop_duplicates(),
        on=["typed", "original"],
        how="left",
    )
    return merged


def analyze_false_negatives(
    input_path: Path,
    output_path: Path,
    *,
    spacy_model: str = SPACY_MODEL,
    device_repo: str | None = None,
    device_split: str = "test",
    diff_path: Path | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    from sage.evaluation.ruspelleval import evaluation as ruspelleval_evaluation

    df = load_predictions(input_path)
    scorer = _load_scorer(spacy_model)

    rows: list[dict[str, Any]] = []
    typed_list = df["typed"].astype(str).tolist()
    original_list = df["original"].astype(str).tolist()
    prediction_list = df["prediction"].astype(str).tolist()

    for sent_id, (typed, original, prediction) in enumerate(
        tqdm(
            zip(typed_list, original_list, prediction_list),
            total=len(df),
            desc="Analyzing FN edits",
        )
    ):
        if not prediction.strip():
            prediction = typed

        ref_edits = _processed_edits(scorer, typed, original)
        hyp_edits = _processed_edits(scorer, typed, prediction)
        src_doc = scorer.errant.annotator.parse(typed)

        fn_keys = set(ref_edits) - set(hyp_edits)
        fp_keys = set(hyp_edits) - set(ref_edits)

        for edit_key in sorted(fn_keys, key=lambda item: (item[0], item[1], item[2])):
            start, end, fn_type, fn_gold_fix = edit_key
            rows.append(
                {
                    "sent_id": sent_id,
                    "typed": typed,
                    "original": original,
                    "prediction": prediction,
                    "fn_type": fn_type,
                    "fn_gold_fix": fn_gold_fix,
                    "source_span": _span_text(src_doc, start, end),
                    "span_start": start,
                    "span_end": end,
                    "num_gold_edits": len(ref_edits),
                    "num_pred_edits": len(hyp_edits),
                    "num_fn_in_sent": len(fn_keys),
                    "num_fp_in_sent": len(fp_keys),
                    "exact_match": prediction == original,
                    "tags": tag_fn_edit(
                        fn_type=fn_type,
                        fn_gold_fix=fn_gold_fix,
                        typed=typed,
                        original=original,
                        prediction=prediction,
                        num_gold_edits=len(ref_edits),
                        num_fn_in_sent=len(fn_keys),
                    ),
                }
            )

    fn_df = pd.DataFrame(rows)
    if not fn_df.empty:
        fn_df["pattern"] = fn_df["fn_type"] + " :: " + fn_df["tags"]

    fn_df = _maybe_merge_device(fn_df, device_repo, device_split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fn_df.to_csv(output_path, index=False)

    print(f"Loaded {len(df)} predictions from {input_path.resolve()}")
    print(f"Found {len(fn_df)} false-negative edits")
    print(f"Saved analysis to {output_path.resolve()}")

    if fn_df.empty:
        return fn_df

    print("\nFN by ERRANT type:")
    for fn_type, count in fn_df["fn_type"].value_counts().items():
        print(f"  {fn_type}: {count}")

    tag_counts = Counter(
        tag for tags in fn_df["tags"] for tag in tags.split("|") if tag
    )
    print("\nFN by heuristic tag:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag}: {count}")

    print(f"\nTop {top_k} FN patterns:")
    for pattern, count in fn_df["pattern"].value_counts().head(top_k).items():
        print(f"  {count:4d}  {pattern}")

    if "device_type" in fn_df.columns and fn_df["device_type"].notna().any():
        print("\nFN by device_type x fn_type:")
        grouped = (
            fn_df.groupby(["device_type", "fn_type"], dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        for (device_type, fn_type), count in grouped.items():
            print(f"  {device_type} / {fn_type}: {count}")

    if diff_path is not None:
        ruspelleval_evaluation(
            typed_list,
            original_list,
            prediction_list,
            to_output_differences=True,
            path_to_diff=str(diff_path),
        )
        print(f"\nSAGE diff saved to {diff_path.resolve()}")

    return fn_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализ false-negative правок spellcheck через ERRANT (SAGE)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV с колонками {REQUIRED_COLUMNS} (по умолчанию: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV с FN-правками (по умолчанию: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Сколько top FN patterns вывести в консоль",
    )
    parser.add_argument(
        "--spacy-model",
        default=SPACY_MODEL,
        help=f"spaCy-модель для ERRANT (по умолчанию: {SPACY_MODEL})",
    )
    parser.add_argument(
        "--device-repo",
        default=None,
        help="HF dataset repo для join device_type (например BW/RU_SPELLCHECK_DEVICE)",
    )
    parser.add_argument(
        "--device-split",
        default="test",
        help="Split HF dataset для join device_type",
    )
    parser.add_argument(
        "--diff",
        type=Path,
        default=None,
        help="Опционально сохранить SAGE word-level diff в текстовый файл",
    )
    args = parser.parse_args()
    analyze_false_negatives(
        args.input,
        args.output,
        spacy_model=args.spacy_model,
        device_repo=args.device_repo,
        device_split=args.device_split,
        diff_path=args.diff,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

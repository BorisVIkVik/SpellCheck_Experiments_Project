"""Evaluate spellcheck predictions with ERRANT (and optional RUSpellEval) via SAGE."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ("typed", "original", "prediction")
DEFAULT_INPUT = Path("prompt_eval_predictions.csv")
SPACY_MODEL = "ru_core_news_lg"


def _ensure_sage_on_path() -> None:
    project_root = Path(__file__).resolve().parent
    sage_root = project_root / "sage"
    if sage_root.is_dir():
        parent = str(sage_root.resolve().parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Файл с предсказаниями не найден: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"В {path} отсутствуют обязательные колонки: {missing}. "
            f"Ожидаются: {list(REQUIRED_COLUMNS)}"
        )
    return df.dropna(subset=list(REQUIRED_COLUMNS))


def tokenize_spellcheck(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def extract_edits(source: str, target: str) -> dict[tuple[int, int], tuple[str, ...]]:
    src_tokens = tokenize_spellcheck(source)
    tgt_tokens = tokenize_spellcheck(target)
    edits: dict[tuple[int, int], tuple[str, ...]] = {}
    matcher = difflib.SequenceMatcher(None, src_tokens, tgt_tokens)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        replacement = tuple(tgt_tokens[j1:j2])
        if src_tokens[i1:i2] != list(replacement):
            edits[(i1, i2)] = replacement
    return edits


def extract_edits_corpus(
    sources: Iterable[str], targets: Iterable[str]
) -> dict[tuple[int, int, int], tuple[str, ...]]:
    corpus_edits: dict[tuple[int, int, int], tuple[str, ...]] = {}
    for sent_id, (source, target) in enumerate(zip(sources, targets)):
        for (i, j), replacement in extract_edits(source, target).items():
            corpus_edits[(sent_id, i, j)] = replacement
    return corpus_edits


def spellcheck_precision_recall(
    sources: Iterable[str],
    corrections: Iterable[str],
    predictions: Iterable[str],
) -> dict[str, float]:
    sources = list(sources)
    corrections = list(corrections)
    predictions = list(predictions)
    if not (len(sources) == len(corrections) == len(predictions)):
        raise ValueError("typed, original и prediction должны быть одной длины")

    gold_edits = extract_edits_corpus(sources, corrections)
    pred_edits = extract_edits_corpus(sources, predictions)
    tp = sum(1 for key, pred_val in pred_edits.items() if gold_edits.get(key) == pred_val)

    n_pred = len(pred_edits)
    n_gold = len(gold_edits)
    precision = tp / n_pred if n_pred else 1.0
    recall = tp / n_gold if n_gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": tp,
        "num_gold_edits": n_gold,
        "num_pred_edits": n_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
    }


def _ensure_spacy_model(metrics: list[str], model_name: str = SPACY_MODEL) -> None:
    if "errant" not in metrics:
        return

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "Для ERRANT нужен spacy. Установите зависимости: "
            "pip install -e \"./sage[errant]\""
        ) from exc

    try:
        spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"Не найдена spaCy-модель '{model_name}'. "
            f"Установите её командой: python -m spacy download {model_name}"
        ) from exc


def spellcheck_metrics_sage(
    sources: Iterable[str],
    corrections: Iterable[str],
    predictions: Iterable[str],
    metrics: list[str],
) -> dict[str, float]:
    _ensure_spacy_model(metrics)

    _ensure_sage_on_path()
    from sage.evaluation import Scorer

    scorer = Scorer(load_errant="errant" in metrics)
    return scorer.score(sources, corrections, predictions, metrics=metrics)


def evaluate_predictions(
    input_path: Path,
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    metrics = metrics or ["errant", "ruspelleval"]
    df = load_predictions(input_path)

    sources = df["typed"].tolist()
    corrections = df["original"].tolist()
    predictions = df["prediction"].tolist()

    exact_match = sum(p == c for p, c in zip(predictions, corrections))
    summary = {
        "input_file": str(input_path.resolve()),
        "num_examples": len(df),
        "exact_match": round(exact_match / len(df) * 100, 2),
        "exact_match_count": exact_match,
    }

    print(f"Loaded {len(df)} predictions from {input_path.resolve()}")
    print(f"Exact match: {summary['exact_match']}% ({exact_match}/{len(df)})")

    sage_metrics = spellcheck_metrics_sage(sources, corrections, predictions, metrics)
    edit_metrics = spellcheck_precision_recall(sources, corrections, predictions)

    print("\nSAGE metrics:")
    for key, value in sorted(sage_metrics.items()):
        print(f"  {key}: {value}")

    print("\nEdit-level metrics:")
    for key, value in sorted(edit_metrics.items()):
        print(f"  {key}: {value}")

    return {"summary": summary, "sage": sage_metrics, "edit_level": edit_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка предсказаний spellcheck через ERRANT / RUSpellEval (SAGE)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV с колонками {REQUIRED_COLUMNS} (по умолчанию: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["errant", "ruspelleval"],
        choices=["errant", "ruspelleval"],
        help="Метрики SAGE для расчёта",
    )
    args = parser.parse_args()
    evaluate_predictions(args.input, metrics=args.metrics)


if __name__ == "__main__":
    main()

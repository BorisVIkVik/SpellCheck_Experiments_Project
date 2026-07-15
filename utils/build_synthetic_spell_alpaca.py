"""Build Alpaca dataset: all BW/spellcheck_benchmark_actualized + synthetic SPELL.

Synthetic part uses SBSC + CharAug with stratified sampling by edit count.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm.auto import tqdm

_UTILS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _UTILS_DIR.parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from convert_spellcheck_benchmark_to_alpaca import (  # noqa: E402
    DEFAULT_STORAGE_FORMAT,
    convert_row,
    load_instruction,
    push_dataset,
)
from evaluate_prompt_errant import extract_edits  # noqa: E402

BENCHMARK_REPO = "BW/spellcheck_benchmark_actualized"
DEVICE_REPO = "BW/RU_SPELLCHECK_DEVICE"
TARGET_REPO = "BW/spellcheck_synthetic_alpaca"
DEFAULT_COUNT = 2000
DEFAULT_SEED = 42
DEFAULT_SPLIT = "train"
DEFAULT_SBSC_RATIO = 0.4
DEFAULT_MAX_ATTEMPTS = 20
DEFAULT_MIN_CLEAN_LEN = 15
DEFAULT_MAX_EDITS = 12

BucketName = Literal["light", "medium", "heavy", "very_heavy"]
MethodName = Literal["sbsc", "char_aug"]

PROMPT_CANDIDATES = [
    _PROJECT_ROOT / "prompt.txt",
    Path("prompt.txt"),
    Path("/content/prompt.txt"),
    Path("/content/NLP_PROJECT/prompt.txt"),
]


@dataclass(frozen=True)
class EditBucket:
    name: BucketName
    min_edits: int
    max_edits: int
    quota: float


BUCKETS: tuple[EditBucket, ...] = (
    EditBucket("light", 1, 1, 0.15),
    EditBucket("medium", 2, 3, 0.35),
    EditBucket("heavy", 4, 6, 0.35),
    EditBucket("very_heavy", 7, DEFAULT_MAX_EDITS, 0.15),
)


def resolve_instruction_file(path: str | Path | None) -> Path | None:
    if path is not None:
        prompt_path = Path(path)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Файл с промптом не найден: {prompt_path}")
        return prompt_path
    return next((p for p in PROMPT_CANDIDATES if p.is_file()), None)


def count_edits(typed: str, clean: str) -> int:
    return len(extract_edits(typed, clean))


def load_clean_sentences(
    benchmark_repo: str,
    device_repo: str,
    split: str,
    seed: int,
) -> list[str]:
    benchmark = load_dataset(benchmark_repo, split=split)
    device = load_dataset(device_repo, split=split)

    benchmark_clean = [
        text.strip()
        for text in tqdm(
            benchmark["correction"],
            desc="Loading benchmark clean sentences",
            leave=False,
        )
        if isinstance(text, str) and len(text.strip()) >= DEFAULT_MIN_CLEAN_LEN
    ]
    device_clean = [
        text.strip()
        for text in tqdm(
            device["original"],
            desc="Loading device clean sentences",
            leave=False,
        )
        if isinstance(text, str) and len(text.strip()) >= DEFAULT_MIN_CLEAN_LEN
    ]

    if not benchmark_clean:
        raise ValueError(f"Не найдено clean-предложений в {benchmark_repo} ({split}).")
    if not device_clean:
        raise ValueError(f"Не найдено clean-предложений в {device_repo} ({split}).")

    rng = random.Random(seed)
    merged = benchmark_clean + device_clean
    rng.shuffle(merged)
    return merged


def build_sbsc_corruptor(device_repo: str, device_split: str, seed: int):
    from sage.spelling_corruption import SBSCConfig, SBSCCorruptor
    from sage.spelling_corruption.sbsc.labeler import process_mistypings

    device = load_dataset(device_repo, split=device_split)
    sources = []
    corrections = []
    for row in tqdm(device, desc="Reading device pairs for SBSC", leave=False):
        sources.append(row["typed"])
        corrections.append(row["original"])
    stats, confusion_matrix, typos_count = process_mistypings(sources, corrections)

    config = SBSCConfig(
        lang="rus",
        typos_count=typos_count,
        stats=stats,
        confusion_matrix=confusion_matrix,
        random_seed=seed,
    )
    return SBSCCorruptor.from_config(config)


def build_char_corruptor(bucket: EditBucket, seed: int):
    from sage.spelling_corruption import CharAugConfig, CharAugCorruptor

    if bucket.name == "light":
        config = CharAugConfig(
            unit_prob=0.15, min_aug=1, max_aug=2, mult_num=2, random_seed=seed
        )
    elif bucket.name == "medium":
        config = CharAugConfig(
            unit_prob=0.25, min_aug=2, max_aug=4, mult_num=3, random_seed=seed
        )
    elif bucket.name == "heavy":
        config = CharAugConfig(
            unit_prob=0.40, min_aug=4, max_aug=8, mult_num=4, random_seed=seed
        )
    else:
        config = CharAugConfig(
            unit_prob=0.50, min_aug=6, max_aug=10, mult_num=4, random_seed=seed
        )
    return CharAugCorruptor.from_config(config)


def bucket_counts(total: int) -> dict[BucketName, int]:
    raw = {bucket.name: int(total * bucket.quota) for bucket in BUCKETS}
    assigned = sum(raw.values())
    if assigned < total:
        raw["medium"] += total - assigned
    elif assigned > total:
        raw["medium"] -= assigned - total
    return raw


def pick_method(rng: random.Random, sbsc_ratio: float) -> MethodName:
    return "sbsc" if rng.random() < sbsc_ratio else "char_aug"


def corrupt_sentence(
    clean: str,
    *,
    bucket: EditBucket,
    method: MethodName,
    sbsc_corruptor,
    char_corruptor,
) -> str:
    if method == "sbsc":
        return sbsc_corruptor.corrupt(clean)
    return char_corruptor.corrupt(clean)


def try_generate_example(
    clean: str,
    *,
    bucket: EditBucket,
    instruction: str,
    sbsc_corruptor,
    char_corruptor,
    rng: random.Random,
    sbsc_ratio: float,
    max_attempts: int,
) -> dict[str, str] | None:
    for _ in range(max_attempts):
        method = pick_method(rng, sbsc_ratio)
        typed = corrupt_sentence(
            clean,
            bucket=bucket,
            method=method,
            sbsc_corruptor=sbsc_corruptor,
            char_corruptor=char_corruptor,
        )
        if not typed or typed.strip() == clean.strip():
            continue

        edits = count_edits(typed, clean)
        if bucket.min_edits <= edits <= bucket.max_edits:
            return {
                "instruction": instruction,
                "input": typed,
                "output": clean,
            }
    return None


def load_benchmark_alpaca(
    benchmark_repo: str,
    instruction: str,
    splits: list[str] | None = None,
) -> Dataset:
    """Load all (or selected) benchmark splits and convert to Alpaca format."""
    raw = load_dataset(benchmark_repo)
    split_names = splits or list(raw.keys())
    missing = [name for name in split_names if name not in raw]
    if missing:
        raise ValueError(
            f"Split(s) {missing} не найдены в {benchmark_repo}. "
            f"Доступны: {list(raw.keys())}"
        )

    parts: list[Dataset] = []
    for split_name in split_names:
        split = raw[split_name]
        parts.append(
            split.map(
                lambda row: convert_row(row, instruction=instruction),
                remove_columns=split.column_names,
                desc=f"Converting benchmark {split_name} to Alpaca",
            )
        )
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def build_synthetic_rows(
    instruction: str,
    *,
    count: int = DEFAULT_COUNT,
    benchmark_repo: str = BENCHMARK_REPO,
    device_repo: str = DEVICE_REPO,
    source_split: str = DEFAULT_SPLIT,
    device_stats_split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    sbsc_ratio: float = DEFAULT_SBSC_RATIO,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Dataset:
    rng = random.Random(seed)
    tqdm.write("Loading clean sentences for synthetic generation...")
    clean_sentences = load_clean_sentences(
        benchmark_repo=benchmark_repo,
        device_repo=device_repo,
        split=source_split,
        seed=seed,
    )
    tqdm.write(f"Loaded {len(clean_sentences)} clean sentences.")

    tqdm.write("Fitting SBSC stats from device train...")
    sbsc_corruptor = build_sbsc_corruptor(
        device_repo=device_repo,
        device_split=device_stats_split,
        seed=seed,
    )
    char_corruptors = {
        bucket.name: build_char_corruptor(bucket, seed + idx)
        for idx, bucket in enumerate(
            tqdm(BUCKETS, desc="Init CharAug corruptors", leave=False)
        )
    }

    rows: list[dict[str, str]] = []
    per_bucket = bucket_counts(count)
    bucket_by_name = {bucket.name: bucket for bucket in BUCKETS}

    with tqdm(total=count, desc="Generating synthetic", unit="ex", dynamic_ncols=True) as overall_pbar:
        for bucket_name, target_count in per_bucket.items():
            bucket = bucket_by_name[bucket_name]
            char_corruptor = char_corruptors[bucket_name]
            generated = 0
            sentence_idx = 0

            with tqdm(
                total=target_count,
                desc=f"  {bucket_name}",
                unit="ex",
                leave=False,
                dynamic_ncols=True,
            ) as bucket_pbar:
                while generated < target_count:
                    clean = clean_sentences[sentence_idx % len(clean_sentences)]
                    sentence_idx += 1

                    example = try_generate_example(
                        clean,
                        bucket=bucket,
                        instruction=instruction,
                        sbsc_corruptor=sbsc_corruptor,
                        char_corruptor=char_corruptor,
                        rng=rng,
                        sbsc_ratio=sbsc_ratio,
                        max_attempts=max_attempts,
                    )
                    if example is None:
                        if sentence_idx > len(clean_sentences) * max_attempts:
                            raise RuntimeError(
                                f"Не удалось набрать {target_count} примеров для bucket={bucket_name}. "
                                f"Сгенерировано только {generated}."
                            )
                        continue

                    rows.append(example)
                    generated += 1
                    bucket_pbar.update(1)
                    overall_pbar.update(1)
                    overall_pbar.set_postfix(bucket=bucket_name, attempts=sentence_idx)

    return Dataset.from_list(rows)


def build_combined_dataset(
    instruction: str,
    *,
    count: int = DEFAULT_COUNT,
    benchmark_repo: str = BENCHMARK_REPO,
    device_repo: str = DEVICE_REPO,
    source_split: str = DEFAULT_SPLIT,
    device_stats_split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    sbsc_ratio: float = DEFAULT_SBSC_RATIO,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    include_benchmark: bool = True,
    benchmark_splits: list[str] | None = None,
) -> tuple[DatasetDict, dict[str, int]]:
    parts: list[Dataset] = []
    composition: dict[str, int] = {}

    if include_benchmark:
        tqdm.write(f"Loading benchmark from {benchmark_repo}...")
        benchmark_ds = load_benchmark_alpaca(
            benchmark_repo=benchmark_repo,
            instruction=instruction,
            splits=benchmark_splits,
        )
        composition["benchmark"] = len(benchmark_ds)
        tqdm.write(f"Benchmark rows: {composition['benchmark']}")
        parts.append(benchmark_ds)

    if count > 0:
        synthetic_ds = build_synthetic_rows(
            instruction,
            count=count,
            benchmark_repo=benchmark_repo,
            device_repo=device_repo,
            source_split=source_split,
            device_stats_split=device_stats_split,
            seed=seed,
            sbsc_ratio=sbsc_ratio,
            max_attempts=max_attempts,
        )
        composition["synthetic"] = len(synthetic_ds)
        tqdm.write(f"Synthetic rows: {composition['synthetic']}")
        parts.append(synthetic_ds)

    if not parts:
        raise ValueError("Датасет пуст: включите benchmark и/или задайте count > 0.")

    train = concatenate_datasets(parts).shuffle(seed=seed)
    composition["total"] = len(train)
    return DatasetDict({"train": train}), composition


def build_synthetic_dataset(
    instruction: str,
    *,
    count: int = DEFAULT_COUNT,
    benchmark_repo: str = BENCHMARK_REPO,
    device_repo: str = DEVICE_REPO,
    source_split: str = DEFAULT_SPLIT,
    device_stats_split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    sbsc_ratio: float = DEFAULT_SBSC_RATIO,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> DatasetDict:
    dataset, _ = build_combined_dataset(
        instruction,
        count=count,
        benchmark_repo=benchmark_repo,
        device_repo=device_repo,
        source_split=source_split,
        device_stats_split=device_stats_split,
        seed=seed,
        sbsc_ratio=sbsc_ratio,
        max_attempts=max_attempts,
        include_benchmark=False,
    )
    return dataset


def print_dataset_stats(
    dataset: DatasetDict,
    composition: dict[str, int] | None = None,
) -> None:
    train = dataset["train"]
    print(f"train: {len(train)} rows")
    if composition:
        print("  composition:")
        for key in ("benchmark", "synthetic", "total"):
            if key in composition:
                print(f"    {key}: {composition[key]}")
    print(f"  columns: {train.column_names}")

    edit_counts = [
        count_edits(row["input"], row["output"])
        for row in tqdm(train, desc="Computing edit stats", unit="ex", dynamic_ncols=True)
    ]
    bucket_hits = {bucket.name: 0 for bucket in BUCKETS}
    for edits in edit_counts:
        for bucket in BUCKETS:
            if bucket.min_edits <= edits <= bucket.max_edits:
                bucket_hits[bucket.name] += 1
                break

    print("  edit count stats:")
    print(f"    min={min(edit_counts)}, max={max(edit_counts)}, avg={sum(edit_counts)/len(edit_counts):.2f}")
    for bucket in BUCKETS:
        print(f"    {bucket.name}: {bucket_hits[bucket.name]}")

    example = train[0]
    print(f"  example input:  {example['input'][:100]}...")
    print(f"  example output: {example['output'][:100]}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Собрать BW/spellcheck_synthetic_alpaca: все данные "
            "BW/spellcheck_benchmark_actualized + synthetic SPELL (SBSC + CharAug)."
        )
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Не включать BW/spellcheck_benchmark_actualized (только synthetic)",
    )
    parser.add_argument(
        "--benchmark-splits",
        nargs="+",
        default=None,
        help="Splits benchmark для включения (по умолчанию: все доступные)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Число synthetic примеров (по умолчанию: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--benchmark-repo",
        default=BENCHMARK_REPO,
        help=f"Источник clean-текстов benchmark (по умолчанию: {BENCHMARK_REPO})",
    )
    parser.add_argument(
        "--device-repo",
        default=DEVICE_REPO,
        help=f"Источник clean-текстов и SBSC stats (по умолчанию: {DEVICE_REPO})",
    )
    parser.add_argument(
        "--source-split",
        default=DEFAULT_SPLIT,
        help=f"Split для clean source sentences (по умолчанию: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--device-stats-split",
        default=DEFAULT_SPLIT,
        help=f"Split для обучения SBSC stats (по умолчанию: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--sbsc-ratio",
        type=float,
        default=DEFAULT_SBSC_RATIO,
        help=f"Доля SBSC среди corruption attempts (по умолчанию: {DEFAULT_SBSC_RATIO})",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Макс. попыток на один clean sentence (по умолчанию: {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed (по умолчанию: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--instruction-file",
        metavar="PATH",
        help="Путь к .txt с instruction (по умолчанию: prompt.txt из репозитория)",
    )
    parser.add_argument(
        "--target-repo",
        default=TARGET_REPO,
        help=f"Целевой репозиторий на HF (по умолчанию: {TARGET_REPO})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (иначе HF_TOKEN / HUGGING_FACE_HUB_TOKEN)",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Загрузить результат на Hugging Face Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Создать приватный репозиторий при загрузке",
    )
    parser.add_argument(
        "--save-local",
        default=None,
        metavar="PATH",
        help="Сохранить датасет локально",
    )
    parser.add_argument(
        "--storage-format",
        choices=["csv", "parquet"],
        default=DEFAULT_STORAGE_FORMAT,
        help=f"Формат файлов при --push (по умолчанию: {DEFAULT_STORAGE_FORMAT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_path = resolve_instruction_file(args.instruction_file)
    if prompt_path is None:
        raise FileNotFoundError(
            "prompt.txt не найден. Передайте --instruction-file или положите prompt.txt в корень проекта."
        )

    instruction = load_instruction(prompt_path)
    print(f"Instruction: {prompt_path.resolve()} ({len(instruction)} chars)")
    print(
        f"Building dataset: benchmark={'yes' if not args.no_benchmark else 'no'}, "
        f"synthetic={args.count}, sbsc_ratio={args.sbsc_ratio}, "
        f"source={args.benchmark_repo}+{args.device_repo} ({args.source_split}), "
        f"SBSC stats from {args.device_repo} ({args.device_stats_split})"
    )

    dataset, composition = build_combined_dataset(
        instruction,
        count=args.count,
        benchmark_repo=args.benchmark_repo,
        device_repo=args.device_repo,
        source_split=args.source_split,
        device_stats_split=args.device_stats_split,
        seed=args.seed,
        sbsc_ratio=args.sbsc_ratio,
        max_attempts=args.max_attempts,
        include_benchmark=not args.no_benchmark,
        benchmark_splits=args.benchmark_splits,
    )
    print_dataset_stats(dataset, composition=composition)

    if args.save_local:
        dataset.save_to_disk(args.save_local)
        print(f"Saved locally to: {args.save_local}")

    if args.push:
        readme_parts = []
        if "benchmark" in composition:
            readme_parts.append(f"{composition['benchmark']} rows from {args.benchmark_repo}")
        if "synthetic" in composition:
            readme_parts.append(
                f"{composition['synthetic']} synthetic SPELL rows (SBSC + CharAug)"
            )
        readme_description = (
            "Russian spellcheck Alpaca dataset. "
            + "; ".join(readme_parts)
            + "."
        )
        print(f"Pushing dataset to: {args.target_repo} ({args.storage_format})")
        push_dataset(
            dataset,
            target_repo=args.target_repo,
            private=args.private,
            token=args.token,
            storage_format=args.storage_format,
            readme_description=readme_description,
        )
        print("Upload complete.")
    elif not args.save_local:
        print("Build finished. Use --push to upload or --save-local to save.")


if __name__ == "__main__":
    main()

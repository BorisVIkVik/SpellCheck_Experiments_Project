"""Build mixed Alpaca dataset for Cloud.ru fine-tuning.

Creates BW/spellcheck_benchmark_alpaca with:
- N benchmark examples sampled from BW/spellcheck_benchmark_actualized (train split)
- M oversampled examples from BW/RU_SPELLCHECK_DEVICE (train split)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from convert_spellcheck_benchmark_to_alpaca import (  # noqa: E402
    TARGET_REPO,
    convert_row,
    load_instruction,
    push_dataset,
)

BENCHMARK_REPO = "BW/spellcheck_benchmark_actualized"
DEVICE_REPO = "BW/RU_SPELLCHECK_DEVICE"
DEFAULT_BENCHMARK_COUNT = 5000
DEFAULT_DEVICE_COUNT = 1000
DEFAULT_SEED = 42
DEFAULT_SPLIT = "train"

PROMPT_CANDIDATES = [
    _UTILS_DIR.parent / "prompt.txt",
    Path("prompt.txt"),
    Path("/content/prompt.txt"),
    Path("/content/NLP_PROJECT/prompt.txt"),
]


def resolve_instruction_file(path: str | Path | None) -> Path | None:
    if path is not None:
        prompt_path = Path(path)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Файл с промптом не найден: {prompt_path}")
        return prompt_path
    return next((p for p in PROMPT_CANDIDATES if p.is_file()), None)


def device_row_to_alpaca(
    row: dict[str, Any],
    instruction: str,
) -> dict[str, str]:
    return {
        "instruction": instruction,
        "input": row["typed"],
        "output": row["original"],
    }


def sample_benchmark_split(
    repo: str,
    split: str,
    count: int,
    seed: int,
    instruction: str,
) -> Dataset:
    benchmark = load_dataset(repo, split=split)
    if count > len(benchmark):
        raise ValueError(
            f"Запрошено {count} примеров из {repo} ({split}), "
            f"но доступно только {len(benchmark)}."
        )

    sampled = benchmark.shuffle(seed=seed).select(range(count))
    return sampled.map(
        lambda row: convert_row(row, instruction=instruction),
        remove_columns=sampled.column_names,
        desc=f"Converting benchmark {split} to Alpaca",
    )


def oversample_device_split(
    repo: str,
    split: str,
    count: int,
    seed: int,
    instruction: str,
) -> Dataset:
    device = load_dataset(repo, split=split)
    if len(device) == 0:
        raise ValueError(f"Split {split} в {repo} пуст.")

    shuffled = device.shuffle(seed=seed)
    indices = [i % len(shuffled) for i in range(count)]
    sampled = shuffled.select(indices)
    return sampled.map(
        lambda row: device_row_to_alpaca(row, instruction=instruction),
        remove_columns=sampled.column_names,
        desc=f"Converting RU_SPELLCHECK_DEVICE {split} to Alpaca",
    )


def build_mixed_dataset(
    instruction: str,
    benchmark_count: int = DEFAULT_BENCHMARK_COUNT,
    device_count: int = DEFAULT_DEVICE_COUNT,
    benchmark_repo: str = BENCHMARK_REPO,
    device_repo: str = DEVICE_REPO,
    benchmark_split: str = DEFAULT_SPLIT,
    device_split: str = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> DatasetDict:
    benchmark_ds = sample_benchmark_split(
        repo=benchmark_repo,
        split=benchmark_split,
        count=benchmark_count,
        seed=seed,
        instruction=instruction,
    )
    device_ds = oversample_device_split(
        repo=device_repo,
        split=device_split,
        count=device_count,
        seed=seed,
        instruction=instruction,
    )

    train = concatenate_datasets([benchmark_ds, device_ds]).shuffle(seed=seed)
    return DatasetDict({"train": train})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Собрать mixed Alpaca-датасет: benchmark train + oversampled "
            "RU_SPELLCHECK_DEVICE train для BW/spellcheck_benchmark_alpaca."
        )
    )
    parser.add_argument(
        "--benchmark-repo",
        default=BENCHMARK_REPO,
        help=f"Источник benchmark (по умолчанию: {BENCHMARK_REPO})",
    )
    parser.add_argument(
        "--device-repo",
        default=DEVICE_REPO,
        help=f"Источник device dataset (по умолчанию: {DEVICE_REPO})",
    )
    parser.add_argument(
        "--benchmark-split",
        default=DEFAULT_SPLIT,
        help=f"Split benchmark (по умолчанию: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--device-split",
        default=DEFAULT_SPLIT,
        help=f"Split device dataset (по умолчанию: {DEFAULT_SPLIT})",
    )
    parser.add_argument(
        "--benchmark-count",
        type=int,
        default=DEFAULT_BENCHMARK_COUNT,
        help=f"Число примеров из benchmark (по умолчанию: {DEFAULT_BENCHMARK_COUNT})",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=DEFAULT_DEVICE_COUNT,
        help=(
            f"Число oversampled примеров из RU_SPELLCHECK_DEVICE "
            f"(по умолчанию: {DEFAULT_DEVICE_COUNT})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed для сэмплирования (по умолчанию: {DEFAULT_SEED})",
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
        f"Building mixed dataset: "
        f"{args.benchmark_count} from {args.benchmark_repo} ({args.benchmark_split}) + "
        f"{args.device_count} oversampled from {args.device_repo} ({args.device_split})"
    )
    dataset = build_mixed_dataset(
        instruction=instruction,
        benchmark_count=args.benchmark_count,
        device_count=args.device_count,
        benchmark_repo=args.benchmark_repo,
        device_repo=args.device_repo,
        benchmark_split=args.benchmark_split,
        device_split=args.device_split,
        seed=args.seed,
    )

    train = dataset["train"]
    print(f"train: {len(train)} rows")
    print(f"  columns: {train.column_names}")
    example = train[0]
    print(f"  example input:  {example['input'][:80]}...")
    print(f"  example output: {example['output'][:80]}...")

    if args.save_local:
        dataset.save_to_disk(args.save_local)
        print(f"Saved locally to: {args.save_local}")

    if args.push:
        print(f"Pushing dataset to: {args.target_repo}")
        push_dataset(
            dataset,
            target_repo=args.target_repo,
            private=args.private,
            token=args.token,
        )
        print("Upload complete.")
    elif not args.save_local:
        print("Build finished. Use --push to upload or --save-local to save.")


if __name__ == "__main__":
    main()

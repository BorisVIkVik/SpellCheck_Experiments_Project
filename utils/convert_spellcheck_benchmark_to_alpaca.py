"""Convert BW/spellcheck_benchmark_actualized to Alpaca format and upload to HF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

SOURCE_REPO = "BW/spellcheck_benchmark_actualized"
TARGET_REPO = "BW/spellcheck_benchmark_alpaca"
DEFAULT_INSTRUCTION = (
    "Исправь орфографические, пунктуационные и регистровые ошибки в русском тексте. "
    "Верни только исправленный текст."
)


def convert_row(
    row: dict[str, Any],
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, str]:
    input_text = row["source"]
    output_text = row["correction"]
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def convert_split(split, instruction: str):
    return split.map(
        lambda row: convert_row(row, instruction=instruction),
        remove_columns=split.column_names,
        desc="Converting to Alpaca format",
    )


def convert_dataset(
    source_repo: str = SOURCE_REPO,
    instruction: str = DEFAULT_INSTRUCTION,
) -> DatasetDict:
    dataset = load_dataset(source_repo)
    return DatasetDict(
        {
            split_name: convert_split(split, instruction=instruction)
            for split_name, split in dataset.items()
        }
    )


def load_instruction(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Файл с промптом не найден: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def resolve_hf_token(token: str | None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def push_dataset(
    dataset: DatasetDict,
    target_repo: str = TARGET_REPO,
    private: bool = False,
    token: str | None = None,
) -> None:
    resolved_token = resolve_hf_token(token)
    if not resolved_token:
        raise ValueError(
            "HF token не задан. Передайте --token или установите HF_TOKEN / HUGGING_FACE_HUB_TOKEN."
        )
    dataset.push_to_hub(
        target_repo,
        private=private,
        token=resolved_token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Скачать BW/spellcheck_benchmark_actualized, конвертировать в Alpaca "
            "и загрузить на Hugging Face как BW/spellcheck_benchmark_alpaca."
        )
    )
    parser.add_argument(
        "--source-repo",
        default=SOURCE_REPO,
        help=f"Исходный датасет на HF (по умолчанию: {SOURCE_REPO})",
    )
    parser.add_argument(
        "--target-repo",
        default=TARGET_REPO,
        help=f"Целевой репозиторий на HF (по умолчанию: {TARGET_REPO})",
    )
    parser.add_argument(
        "--instruction-file",
        metavar="PATH",
        help="Путь к .txt файлу с промптом для поля instruction",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token для загрузки датасета (иначе берётся из HF_TOKEN)",
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
        help="Сохранить датасет локально (например, ./spellcheck_benchmark_alpaca)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    instruction = (
        load_instruction(args.instruction_file)
        if args.instruction_file
        else DEFAULT_INSTRUCTION
    )

    print(f"Loading dataset: {args.source_repo}")
    dataset = convert_dataset(
        source_repo=args.source_repo,
        instruction=instruction,
    )

    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)} rows")
        example = split[0]
        print(f"  columns: {split.column_names}")
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
        print("Conversion finished. Use --push to upload or --save-local to save.")


if __name__ == "__main__":
    main()

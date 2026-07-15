"""Convert BW/spellcheck_benchmark_actualized to Alpaca format and upload to HF."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo

SOURCE_REPO = "BW/spellcheck_benchmark_actualized"
TARGET_REPO = "BW/spellcheck_benchmark_alpaca"
DEFAULT_INSTRUCTION = (
    "Исправь орфографические, пунктуационные и регистровые ошибки в русском тексте. "
    "Верни только исправленный текст."
)
StorageFormat = Literal["csv", "parquet"]
DEFAULT_STORAGE_FORMAT: StorageFormat = "csv"


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


def drop_empty_alpaca_rows(dataset: DatasetDict) -> DatasetDict:
    def keep_row(example: dict[str, str]) -> bool:
        return bool(str(example["input"]).strip()) and bool(str(example["output"]).strip())

    return DatasetDict(
        {
            split_name: split.filter(keep_row, desc=f"Filter empty rows in {split_name}")
            for split_name, split in dataset.items()
        }
    )


def build_dataset_readme(dataset: DatasetDict, *, description: str | None = None) -> str:
    split_lines = []
    data_file_lines = []
    for split_name, split in dataset.items():
        split_lines.append(f"  - name: {split_name}\n    num_examples: {len(split)}")
        data_file_lines.append(f"  - split: {split_name}\n    path: {split_name}.csv")

    body = description or (
        "Russian spell correction dataset in Alpaca format "
        "(`instruction`, `input`, `output`)."
    )
    return (
        "---\n"
        "dataset_info:\n"
        "  features:\n"
        "  - name: instruction\n"
        "    dtype: string\n"
        "  - name: input\n"
        "    dtype: string\n"
        "  - name: output\n"
        "    dtype: string\n"
        "  splits:\n"
        f"{chr(10).join(split_lines)}\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        f"{chr(10).join(data_file_lines)}\n"
        "---\n\n"
        f"{body}\n"
    )


def push_dataset_csv(
    dataset: DatasetDict,
    target_repo: str,
    *,
    private: bool = False,
    token: str | None = None,
    readme_description: str | None = None,
) -> None:
    resolved_token = resolve_hf_token(token)
    if not resolved_token:
        raise ValueError(
            "HF token не задан. Передайте --token или установите HF_TOKEN / HUGGING_FACE_HUB_TOKEN."
        )

    dataset = drop_empty_alpaca_rows(dataset)
    create_repo(
        target_repo,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=resolved_token,
    )

    api = HfApi(token=resolved_token)
    with tempfile.TemporaryDirectory(prefix="alpaca_dataset_") as tmp_dir:
        root = Path(tmp_dir)
        (root / "README.md").write_text(
            build_dataset_readme(dataset, description=readme_description),
            encoding="utf-8",
        )
        for split_name, split in dataset.items():
            split.to_csv(str(root / f"{split_name}.csv"))

        api.upload_folder(
            folder_path=str(root),
            repo_id=target_repo,
            repo_type="dataset",
            commit_message="Upload Alpaca dataset as CSV",
        )


def push_dataset(
    dataset: DatasetDict,
    target_repo: str = TARGET_REPO,
    private: bool = False,
    token: str | None = None,
    storage_format: StorageFormat = DEFAULT_STORAGE_FORMAT,
    readme_description: str | None = None,
) -> None:
    if storage_format == "csv":
        push_dataset_csv(
            dataset,
            target_repo,
            private=private,
            token=token,
            readme_description=readme_description,
        )
        return

    resolved_token = resolve_hf_token(token)
    if not resolved_token:
        raise ValueError(
            "HF token не задан. Передайте --token или установите HF_TOKEN / HUGGING_FACE_HUB_TOKEN."
        )
    dataset = drop_empty_alpaca_rows(dataset)
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
    parser.add_argument(
        "--storage-format",
        choices=["csv", "parquet"],
        default=DEFAULT_STORAGE_FORMAT,
        help=f"Формат файлов при --push (по умолчанию: {DEFAULT_STORAGE_FORMAT})",
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
            storage_format=args.storage_format,
        )
        print("Upload complete.")
    elif not args.save_local:
        print("Conversion finished. Use --push to upload or --save-local to save.")


if __name__ == "__main__":
    main()

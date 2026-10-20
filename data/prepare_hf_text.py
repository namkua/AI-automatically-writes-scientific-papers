"""Download small, real Hugging Face text subsets for local experiments.

Usage:
    .\env\Scripts\python.exe data/prepare_hf_text.py --dataset ag_news
    .\env\Scripts\python.exe data/prepare_hf_text.py --dataset imdb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent / "huggingface"
DATASETS = {
    "ag_news": {"path": "ag_news", "train": "train[:5000]", "test": "test[:1000]"},
    "imdb": {"path": "imdb", "train": "train[:5000]", "test": "test[:1000]"},
}


def export_split(dataset_name: str, split_name: str, output_dir: Path) -> int:
    rows = load_dataset(dataset_name, split=split_name)
    output = output_dir / f"{split_name.split('[')[0]}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small verified Hugging Face text dataset subset.")
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    args = parser.parse_args()
    selected = [args.dataset] if args.dataset else sorted(DATASETS)
    for dataset_name in selected:
        config = DATASETS[dataset_name]
        output_dir = ROOT / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        train_count = export_split(config["path"], config["train"], output_dir)
        test_count = export_split(config["path"], config["test"], output_dir)
        print(f"{dataset_name}: wrote {train_count} train rows and {test_count} test rows to {output_dir}")


if __name__ == "__main__":
    main()

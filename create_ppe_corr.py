#!/usr/bin/env python3
"""Create PPE correctness dataset for Accuracy scoring."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from datasets import load_dataset

BENCHMARKS = [
    {
        "domain": "mmlu_pro",
        "hf_path": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
        "split": "train",
    },
    {
        "domain": "math",
        "hf_path": "lmarena-ai/PPE-MATH-Best-of-K",
        "split": "train",
    },
    {
        "domain": "gpqa",
        "hf_path": "lmarena-ai/PPE-GPQA-Best-of-K",
        "split": "train",
    },
    {
        "domain": "ifeval",
        "hf_path": "lmarena-ai/PPE-IFEval-Best-of-K",
        "split": "train",
    },
    {
        "domain": "mbpp_plus",
        "hf_path": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
        "split": "train",
    },
]


def _build_item(domain: str, row: Dict, index: int) -> Dict:
    prompt = row["prompt"]
    question_id = row.get("question_id")
    item_id = f"{domain}:{question_id}" if question_id is not None else f"{domain}:{index}"

    chosen: List[str] = []
    rejected: List[str] = []

    scores = row["scores"]
    for pair in row["sampled_conflict_pairs"]:
        i, j = pair
        if scores[i] > scores[j]:
            chosen.append(row[f"response_{i + 1}"])
            rejected.append(row[f"response_{j + 1}"])
        else:
            chosen.append(row[f"response_{j + 1}"])
            rejected.append(row[f"response_{i + 1}"])

    return {
        "id": item_id,
        "domain": domain,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def build_dataset() -> List[Dict]:
    records: List[Dict] = []

    for benchmark in BENCHMARKS:
        dataset = load_dataset(benchmark["hf_path"], split=benchmark["split"])
        for index, row in enumerate(dataset):
            records.append(_build_item(benchmark["domain"], row, index))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download PPE correctness benchmarks and build the Accuracy dataset JSON."
        )
    )
    parser.add_argument(
        "--output",
        default="PPE_Corr.json",
        help="Output path for the aggregated PPE correctness dataset JSON.",
    )
    args = parser.parse_args()

    records = build_dataset()

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

"""Build PPE Correctness Accuracy dataset.

This script downloads PPE correctness benchmarks and converts the conflict pairs
into a single JSON file suitable for Accuracy evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import load_dataset


DATASETS: Dict[str, str] = {
    "mmlu_pro": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
    "math": "lmarena-ai/PPE-MATH-Best-of-K",
    "gpqa": "lmarena-ai/PPE-GPQA-Best-of-K",
    "ifeval": "lmarena-ai/PPE-IFEval-Best-of-K",
    "mbpp_plus": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
}

DEFAULT_SPLIT = "train"
OUTPUT_PATH = Path("PPE_Corr.json")


def _row_id(domain: str, row: dict, index: int) -> str:
    question_id = row.get("question_id") or row.get("id") or str(index)
    model_name = row.get("model_name")
    if model_name:
        return f"{domain}:{question_id}:{model_name}"
    return f"{domain}:{question_id}"


def _responses_for_pair(row: dict, left_idx: int, right_idx: int) -> tuple[str, str]:
    left_response = row[f"response_{left_idx + 1}"]
    right_response = row[f"response_{right_idx + 1}"]
    return left_response, right_response


def _build_items() -> Iterable[dict]:
    for domain, dataset_name in DATASETS.items():
        dataset = load_dataset(dataset_name, split=DEFAULT_SPLIT)
        for index, row in enumerate(dataset):
            scores = row["scores"]
            pairs = row["sampled_conflict_pairs"]

            chosen: List[str] = []
            rejected: List[str] = []

            for left_idx, right_idx in pairs:
                left_response, right_response = _responses_for_pair(
                    row, left_idx, right_idx
                )
                if scores[left_idx] > scores[right_idx]:
                    chosen.append(left_response)
                    rejected.append(right_response)
                else:
                    chosen.append(right_response)
                    rejected.append(left_response)

            yield {
                "id": _row_id(domain, row, index),
                "domain": domain,
                "prompt": row["prompt"],
                "chosen": chosen,
                "rejected": rejected,
            }


def main() -> None:
    items = list(_build_items())
    OUTPUT_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(items)} prompts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

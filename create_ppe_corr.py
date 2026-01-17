"""Build PPE Correctness accuracy dataset.

Downloads PPE correctness benchmarks from Hugging Face and constructs a merged
list of chosen/rejected responses per prompt for accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset

BENCHMARKS = [
    {
        "dataset": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
        "domain": "MMLU-Pro",
    },
    {
        "dataset": "lmarena-ai/PPE-MATH-Best-of-K",
        "domain": "MATH",
    },
    {
        "dataset": "lmarena-ai/PPE-GPQA-Best-of-K",
        "domain": "GPQA",
    },
    {
        "dataset": "lmarena-ai/PPE-IFEval-Best-of-K",
        "domain": "IFEval",
    },
    {
        "dataset": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
        "domain": "MBPP-Plus",
    },
]

RESPONSE_KEY_RE = re.compile(r"^response_(\d+)$")


def _response_keys(row: Dict[str, object]) -> List[str]:
    keys: List[Tuple[int, str]] = []
    for key in row.keys():
        match = RESPONSE_KEY_RE.match(key)
        if match:
            keys.append((int(match.group(1)), key))
    return [key for _, key in sorted(keys, key=lambda item: item[0])]


def _extract_responses(row: Dict[str, object]) -> List[str]:
    return [row[key] for key in _response_keys(row)]


def _add_unique(items: List[str], value: str, seen: set) -> None:
    if value not in seen:
        items.append(value)
        seen.add(value)


def _iter_rows(dataset_name: str, split: str) -> Iterable[Dict[str, object]]:
    dataset = load_dataset(dataset_name, split=split)
    for row in dataset:
        yield row


def build_ppe_correctness(split: str) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str], Dict[str, object]] = {}

    for benchmark in BENCHMARKS:
        dataset_name = benchmark["dataset"]
        domain = benchmark["domain"]

        for row_index, row in enumerate(_iter_rows(dataset_name, split=split)):
            prompt = row["prompt"]
            key = (domain, prompt)

            if key not in merged:
                question_id = row.get("question_id")
                if question_id is None:
                    question_id = f"{row_index}"
                merged[key] = {
                    "id": f"{domain}-{question_id}",
                    "domain": domain,
                    "prompt": prompt,
                    "chosen": [],
                    "rejected": [],
                }

            item = merged[key]
            chosen_list: List[str] = item["chosen"]
            rejected_list: List[str] = item["rejected"]
            chosen_seen = set(chosen_list)
            rejected_seen = set(rejected_list)

            responses = _extract_responses(row)
            scores = row["scores"]
            conflict_pairs = row["sampled_conflict_pairs"]

            for pair in conflict_pairs:
                i, j = pair
                score_i = scores[i]
                score_j = scores[j]
                if score_i == score_j:
                    continue
                if score_i > score_j:
                    chosen, rejected = responses[i], responses[j]
                else:
                    chosen, rejected = responses[j], responses[i]
                _add_unique(chosen_list, chosen, chosen_seen)
                _add_unique(rejected_list, rejected, rejected_seen)

    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download PPE correctness benchmarks and build the accuracy dataset."
        )
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--output",
        default="PPE_Corr.json",
        help="Output JSON path (default: PPE_Corr.json).",
    )
    args = parser.parse_args()

    data = build_ppe_correctness(split=args.split)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(data)} prompts to {args.output}.")


if __name__ == "__main__":
    main()

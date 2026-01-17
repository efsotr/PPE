#!/usr/bin/env python3
"""Build PPE correctness accuracy dataset."""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


BENCHMARKS = {
    "mmlu_pro_best_of_k": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
    "math_best_of_k": "lmarena-ai/PPE-MATH-Best-of-K",
    "gpqa_best_of_k": "lmarena-ai/PPE-GPQA-Best-of-K",
    "ifeval_best_of_k": "lmarena-ai/PPE-IFEval-Best-of-K",
    "mbpp_plus_best_of_k": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
}


@dataclass
class PromptBucket:
    domain: str
    prompt: str
    chosen: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    _chosen_seen: set = field(default_factory=set, repr=False)
    _rejected_seen: set = field(default_factory=set, repr=False)

    def add_chosen(self, response: str) -> None:
        if response not in self._chosen_seen:
            self.chosen.append(response)
            self._chosen_seen.add(response)

    def add_rejected(self, response: str) -> None:
        if response not in self._rejected_seen:
            self.rejected.append(response)
            self._rejected_seen.add(response)


def prompt_id(domain: str, prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    return f"{domain}:{digest}"


def response_keys(row: Dict) -> List[str]:
    keys = []
    idx = 1
    while True:
        key = f"response_{idx}"
        if key not in row:
            break
        keys.append(key)
        idx += 1
    return keys


def iter_conflict_pairs(
    scores: List[float],
    pairs: Iterable[Iterable[int]],
    responses: List[str],
) -> Iterable[Tuple[str, str]]:
    for pair in pairs:
        i, j = pair
        if scores[i] == scores[j]:
            continue
        if scores[i] > scores[j]:
            yield responses[i], responses[j]
        else:
            yield responses[j], responses[i]


def build_dataset(split: str) -> List[Dict]:
    buckets: Dict[Tuple[str, str], PromptBucket] = {}

    for domain, dataset_name in BENCHMARKS.items():
        dataset = load_dataset(dataset_name, split=split)
        for row in dataset:
            prompt = row["prompt"]
            key = (domain, prompt)
            if key not in buckets:
                buckets[key] = PromptBucket(domain=domain, prompt=prompt)

            scores = row["scores"]
            pairs = row.get("sampled_conflict_pairs", [])
            resp_keys = response_keys(row)
            responses = [row[key] for key in resp_keys]

            for chosen, rejected in iter_conflict_pairs(scores, pairs, responses):
                buckets[key].add_chosen(chosen)
                buckets[key].add_rejected(rejected)

    records = []
    for (domain, prompt), bucket in buckets.items():
        if not bucket.chosen or not bucket.rejected:
            continue
        records.append(
            {
                "id": prompt_id(domain, prompt),
                "domain": domain,
                "prompt": prompt,
                "chosen": bucket.chosen,
                "rejected": bucket.rejected,
            }
        )

    records.sort(key=lambda item: (item["domain"], item["id"]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create PPE correctness accuracy dataset as PPE_Corr.json."
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--output",
        default="PPE_Corr.json",
        help="Output JSON path (default: PPE_Corr.json).",
    )
    args = parser.parse_args()

    records = build_dataset(args.split)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

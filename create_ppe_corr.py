import argparse
import hashlib
import json
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


BENCHMARKS: Tuple[Tuple[str, str, str], ...] = (
    ("mmlu_pro_best_of_k", "lmarena-ai/PPE-MMLU-Pro-Best-of-K", "mmlu_pro"),
    ("math_best_of_k", "lmarena-ai/PPE-MATH-Best-of-K", "math"),
    ("gpqa_best_of_k", "lmarena-ai/PPE-GPQA-Best-of-K", "gpqa"),
    ("ifeval_best_of_k", "lmarena-ai/PPE-IFEval-Best-of-K", "ifeval"),
    ("mbpp_plus_best_of_k", "lmarena-ai/PPE-MBPP-Plus-Best-of-K", "mbpp_plus"),
)


def get_response_keys(column_names: Iterable[str]) -> List[str]:
    response_keys = [
        name for name in column_names if name.startswith("response_")
    ]
    response_keys.sort(key=lambda name: int(name.split("_")[1]))
    return response_keys


def build_prompt_id(domain: str, prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    return f"{domain}:{digest}"


def add_conflict_pairs(
    items: Dict[Tuple[str, str], Dict[str, object]],
    domain: str,
    prompt: str,
    response_keys: List[str],
    responses: Dict[str, str],
    scores: List[float],
    conflict_pairs: Iterable[Iterable[int]],
) -> None:
    key = (domain, prompt)
    if key not in items:
        items[key] = {
            "id": build_prompt_id(domain, prompt),
            "domain": domain,
            "prompt": prompt,
            "chosen": [],
            "rejected": [],
        }

    chosen_list: List[str] = items[key]["chosen"]
    rejected_list: List[str] = items[key]["rejected"]

    for pair in conflict_pairs:
        left_idx, right_idx = pair
        if scores[left_idx] == scores[right_idx]:
            continue
        if scores[left_idx] > scores[right_idx]:
            chosen_key = response_keys[left_idx]
            rejected_key = response_keys[right_idx]
        else:
            chosen_key = response_keys[right_idx]
            rejected_key = response_keys[left_idx]
        chosen_list.append(responses[chosen_key])
        rejected_list.append(responses[rejected_key])


def load_correctness_data(split: str) -> List[Dict[str, object]]:
    items: Dict[Tuple[str, str], Dict[str, object]] = OrderedDict()

    for _, dataset_path, domain in BENCHMARKS:
        dataset = load_dataset(dataset_path, split=split)
        response_keys = get_response_keys(dataset.column_names)

        for row in dataset:
            prompt = row["prompt"]
            scores = row["scores"]
            conflict_pairs = row["sampled_conflict_pairs"]

            responses = {key: row[key] for key in response_keys}

            add_conflict_pairs(
                items,
                domain,
                prompt,
                response_keys,
                responses,
                scores,
                conflict_pairs,
            )

    return list(items.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download PPE correctness datasets and build PPE_Corr.json for accuracy."
        )
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use for PPE correctness benchmarks (default: train).",
    )
    parser.add_argument(
        "--output",
        default="PPE_Corr.json",
        help="Output path for the merged PPE correctness dataset.",
    )
    args = parser.parse_args()

    data = load_correctness_data(args.split)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

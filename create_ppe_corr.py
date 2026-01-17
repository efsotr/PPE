import argparse
import json
from typing import Dict, List

from datasets import load_dataset


BENCHMARKS = [
    {
        "name": "mmlu_pro_best_of_k",
        "path": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
        "split": "train",
        "responses_per_question": 32,
    },
    {
        "name": "math_best_of_k",
        "path": "lmarena-ai/PPE-MATH-Best-of-K",
        "split": "train",
        "responses_per_question": 32,
    },
    {
        "name": "gpqa_best_of_k",
        "path": "lmarena-ai/PPE-GPQA-Best-of-K",
        "split": "train",
        "responses_per_question": 32,
    },
    {
        "name": "ifeval_best_of_k",
        "path": "lmarena-ai/PPE-IFEval-Best-of-K",
        "split": "train",
        "responses_per_question": 32,
    },
    {
        "name": "mbpp_plus_best_of_k",
        "path": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
        "split": "train",
        "responses_per_question": 32,
    },
]


def build_item(
    row: Dict,
    benchmark_name: str,
    responses_per_question: int,
) -> Dict[str, object]:
    responses = [
        row[f"response_{i + 1}"] for i in range(responses_per_question)
    ]
    scores = row["scores"]
    chosen: List[str] = []
    rejected: List[str] = []

    for pair in row["sampled_conflict_pairs"]:
        i, j = pair
        if scores[i] == scores[j]:
            continue
        if scores[i] > scores[j]:
            chosen.append(responses[i])
            rejected.append(responses[j])
        else:
            chosen.append(responses[j])
            rejected.append(responses[i])

    return {
        "id": f"{benchmark_name}:{row['question_id']}",
        "domain": benchmark_name,
        "prompt": row["prompt"],
        "chosen": chosen,
        "rejected": rejected,
    }


def build_dataset() -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []

    for benchmark in BENCHMARKS:
        dataset = load_dataset(benchmark["path"], split=benchmark["split"])
        for row in dataset:
            items.append(
                build_item(
                    row,
                    benchmark["name"],
                    benchmark["responses_per_question"],
                )
            )

    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create PPE Correctness dataset for Accuracy metric."
    )
    parser.add_argument(
        "--output",
        default="PPE_Corr.json",
        help="Output path for the generated dataset.",
    )
    args = parser.parse_args()

    items = build_dataset()
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(items, output_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

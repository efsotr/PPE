import argparse
import json
from typing import Dict, List

from datasets import load_dataset


DATASETS = {
    "mmlu_pro_best_of_k": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
    "math_best_of_k": "lmarena-ai/PPE-MATH-Best-of-K",
    "gpqa_best_of_k": "lmarena-ai/PPE-GPQA-Best-of-K",
    "ifeval_best_of_k": "lmarena-ai/PPE-IFEval-Best-of-K",
    "mbpp_plus_best_of_k": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
}


def build_prompt_entry(domain: str, row: Dict, fallback_id: str) -> Dict:
    question_id = row.get("question_id", fallback_id)
    prompt = row["prompt"]
    scores = row["scores"]
    chosen: List[str] = []
    rejected: List[str] = []

    for pair in row["sampled_conflict_pairs"]:
        i, j = pair
        if scores[i] > scores[j]:
            chosen.append(row[f"response_{i + 1}"])
            rejected.append(row[f"response_{j + 1}"])
        else:
            chosen.append(row[f"response_{j + 1}"])
            rejected.append(row[f"response_{i + 1}"])

    return {
        "id": f"{domain}:{question_id}",
        "domain": domain,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def build_ppe_correctness(split: str) -> List[Dict]:
    records: List[Dict] = []
    for domain, dataset_name in DATASETS.items():
        dataset = load_dataset(dataset_name, split=split)
        for idx, row in enumerate(dataset):
            records.append(build_prompt_entry(domain, row, fallback_id=str(idx)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PPE correctness dataset for accuracy metric."
    )
    parser.add_argument(
        "--output",
        default="PPE_Corr.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use.",
    )
    args = parser.parse_args()

    records = build_ppe_correctness(args.split)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

import argparse
import json
from typing import Dict, List

from datasets import load_dataset


CORRECTNESS_BENCHMARKS: Dict[str, str] = {
    "mmlu_pro": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
    "math": "lmarena-ai/PPE-MATH-Best-of-K",
    "gpqa": "lmarena-ai/PPE-GPQA-Best-of-K",
    "ifeval": "lmarena-ai/PPE-IFEval-Best-of-K",
    "mbpp_plus": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
}


def _response_columns(column_names: List[str]) -> List[str]:
    return sorted(
        [c for c in column_names if c.startswith("response_")],
        key=lambda name: int(name.split("_")[1]),
    )


def build_records(split: str) -> List[Dict]:
    records: List[Dict] = []

    for domain, dataset_path in CORRECTNESS_BENCHMARKS.items():
        dataset = load_dataset(dataset_path, split=split)

        response_cols = _response_columns(dataset.column_names)

        for idx, row in enumerate(dataset):
            responses = [row[col] for col in response_cols]
            scores = row["scores"]
            pairs = row["sampled_conflict_pairs"]

            chosen, rejected = [], []

            for first, second in pairs:
                if (
                    max(first, second) >= len(responses)
                    or max(first, second) >= len(scores)
                    or scores[first] == scores[second]
                ):
                    continue

                if scores[first] > scores[second]:
                    chosen.append(responses[first])
                    rejected.append(responses[second])
                else:
                    chosen.append(responses[second])
                    rejected.append(responses[first])

            question_id = row.get("question_id") or row.get("id") or row.get("uid")
            if question_id is None:
                question_id = f"{domain}-{idx}"

            records.append(
                {
                    "id": str(question_id),
                    "domain": domain,
                    "prompt": row["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    return records


def main(output_path: str, split: str = "train"):
    records = build_records(split=split)

    with open(output_path, "w", encoding="utf-8") as ofile:
        json.dump(records, ofile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and build PPE Correctness accuracy dataset."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="PPE_Corr.json",
        help="Where to save the aggregated correctness dataset.",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="train",
        help="Dataset split to load from each benchmark.",
    )

    args = parser.parse_args()
    main(output_path=args.output, split=args.split)

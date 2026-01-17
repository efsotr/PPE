import argparse
import json
from typing import Dict, Iterable, List, Sequence

from datasets import load_dataset


BENCHMARKS = [
    {
        "name": "mmlu_pro_best_of_k",
        "hf_path": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
        "split": "train",
        "domain": "mmlu_pro",
    },
    {
        "name": "math_best_of_k",
        "hf_path": "lmarena-ai/PPE-MATH-Best-of-K",
        "split": "train",
        "domain": "math",
    },
    {
        "name": "gpqa_best_of_k",
        "hf_path": "lmarena-ai/PPE-GPQA-Best-of-K",
        "split": "train",
        "domain": "gpqa",
    },
    {
        "name": "ifeval_best_of_k",
        "hf_path": "lmarena-ai/PPE-IFEval-Best-of-K",
        "split": "train",
        "domain": "ifeval",
    },
    {
        "name": "mbpp_plus_best_of_k",
        "hf_path": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
        "split": "train",
        "domain": "mbpp_plus",
    },
]


def _get_response(row: Dict, index: int) -> str:
    key = f"response_{index + 1}"
    if key in row:
        return row[key]
    responses = row.get("responses")
    if responses is None:
        raise KeyError(f"Missing response {index + 1} in row keys: {row.keys()}")
    return responses[index]


def _append_unique(target: List[str], value: str) -> None:
    if value not in target:
        target.append(value)


def _iter_conflict_pairs(row: Dict) -> Iterable[Sequence[int]]:
    pairs = row.get("sampled_conflict_pairs")
    if pairs is None:
        raise KeyError("Row missing sampled_conflict_pairs")
    return pairs


def _build_items(rows: Iterable[Dict], domain: str) -> List[Dict]:
    items = []
    for idx, row in enumerate(rows):
        prompt = row["prompt"]
        question_id = row.get("question_id", f"{domain}-{idx}")
        scores = row["scores"]
        chosen: List[str] = []
        rejected: List[str] = []
        for pair in _iter_conflict_pairs(row):
            i, j = pair
            if scores[i] == scores[j]:
                continue
            if scores[i] > scores[j]:
                _append_unique(chosen, _get_response(row, i))
                _append_unique(rejected, _get_response(row, j))
            else:
                _append_unique(chosen, _get_response(row, j))
                _append_unique(rejected, _get_response(row, i))
        if not chosen or not rejected:
            continue
        items.append(
            {
                "id": f"{domain}:{question_id}",
                "domain": domain,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return items


def _load_rows(hf_path: str, split: str) -> List[Dict]:
    dataset = load_dataset(hf_path, split=split)
    return [dataset[i] for i in range(len(dataset))]


def build_ppe_corr_dataset() -> List[Dict]:
    all_items: List[Dict] = []
    for benchmark in BENCHMARKS:
        rows = _load_rows(benchmark["hf_path"], benchmark["split"])
        all_items.extend(_build_items(rows, benchmark["domain"]))
    return all_items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download PPE correctness benchmarks and build PPE_Corr.json "
            "for accuracy evaluation."
        )
    )
    parser.add_argument(
        "--output",
        default="PPE_Corr.json",
        help="Output path for the merged dataset.",
    )
    args = parser.parse_args()

    items = build_ppe_corr_dataset()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

import argparse
import json

from benchmarks.base import benchmark_registry

CORRECTNESS_BENCHMARKS = {
    "mmlu_pro_best_of_k": "mmlu_pro",
    "math_best_of_k": "math",
    "gpqa_best_of_k": "gpqa",
    "ifeval_best_of_k": "ifeval",
    "mbpp_plus_best_of_k": "mbpp_plus",
}


def get_item_id(row, benchmark_name: str) -> str:
    item_id = row.get("uid")
    if item_id is None:
        item_id = row.get("question_id")
    if item_id is None:
        raise ValueError(f"Missing uid/question_id for {benchmark_name}")
    return item_id


def build_ppe_corr(output_path: str) -> None:
    records = []

    for benchmark_name, domain in CORRECTNESS_BENCHMARKS.items():
        benchmark_cls = benchmark_registry.get(benchmark_name)
        if benchmark_cls is None:
            available = ", ".join(sorted(benchmark_registry.keys()))
            raise ValueError(
                f"Unknown benchmark: {benchmark_name}. Available benchmarks: {available}"
            )

        benchmark = benchmark_cls(iterator=False)

        for _, row in benchmark.get_conflict_pair_iter():
            item_id = get_item_id(row, benchmark_name)
            # ground_truth == 1 indicates response_1 scored higher than response_2.
            response_1_is_preferred = row["ground_truth"] == 1

            if response_1_is_preferred:
                chosen = row["response_1"]
                rejected = row["response_2"]
            else:
                chosen = row["response_2"]
                rejected = row["response_1"]

            records.append(
                {
                    "id": item_id,
                    "domain": domain,
                    "prompt": row["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    try:
        with open(output_path, "w") as output_file:
            json.dump(records, output_file, indent=2)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write output to {output_path}: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PPE_Corr.json from PPE correctness benchmarks."
    )
    parser.add_argument(
        "--output-path", "-o", default="PPE_Corr.json", help="Output JSON path."
    )
    args = parser.parse_args()
    build_ppe_corr(args.output_path)


if __name__ == "__main__":
    main()

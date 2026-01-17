"""
Script to download and build PPE Correctness dataset for Accuracy metric calculation.

This script downloads the 5 Correctness Preference Benchmarks and constructs
a dataset with chosen vs rejected response pairs based on correctness labels.
"""

import json
from collections import defaultdict

from datasets import load_dataset


# Correctness benchmark configurations
CORRECTNESS_BENCHMARKS = [
    {
        "name": "mmlu_pro_best_of_k",
        "path": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
        "domain": "mmlu_pro",
    },
    {
        "name": "math_best_of_k",
        "path": "lmarena-ai/PPE-MATH-Best-of-K",
        "domain": "math",
    },
    {
        "name": "gpqa_best_of_k",
        "path": "lmarena-ai/PPE-GPQA-Best-of-K",
        "domain": "gpqa",
    },
    {
        "name": "ifeval_best_of_k",
        "path": "lmarena-ai/PPE-IFEval-Best-of-K",
        "domain": "ifeval",
    },
    {
        "name": "mbpp_plus_best_of_k",
        "path": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
        "domain": "mbpp_plus",
    },
]


def _make_prompt_entry():
    """Factory function for defaultdict to create new prompt entries."""
    return {"id": None, "chosen": [], "rejected": []}


def process_benchmark(benchmark_config: dict) -> list:
    """
    Process a single benchmark and extract conflict pairs.

    Args:
        benchmark_config: Configuration dict with name, path, and domain.

    Returns:
        List of items with id, domain, prompt, chosen, and rejected lists.
    """
    print(f"Loading {benchmark_config['name']}...")
    ds = load_dataset(benchmark_config["path"], split="train")

    # Group by prompt to collect all chosen/rejected pairs
    # Use prompt text as key, store first question_id seen for each prompt
    prompt_data = defaultdict(_make_prompt_entry)

    for row in ds:
        question_id = row["question_id"]
        prompt = row["prompt"]
        scores = row["scores"]
        pairs = row["sampled_conflict_pairs"]

        # Use prompt text as key for grouping
        if prompt_data[prompt]["id"] is None:
            prompt_data[prompt]["id"] = question_id

        for pair in pairs:
            i, j = pair[0], pair[1]
            # scores[i] > scores[j] means response i is correct (chosen)
            # and response j is incorrect (rejected)
            if scores[i] > scores[j]:
                chosen_idx = i
                rejected_idx = j
            else:
                chosen_idx = j
                rejected_idx = i

            chosen_response = row[f"response_{chosen_idx + 1}"]
            rejected_response = row[f"response_{rejected_idx + 1}"]

            prompt_data[prompt]["chosen"].append(chosen_response)
            prompt_data[prompt]["rejected"].append(rejected_response)

    # Convert to final format
    results = []
    for prompt, data in prompt_data.items():
        item = {
            "id": data["id"],
            "domain": benchmark_config["domain"],
            "prompt": prompt,
            "chosen": data["chosen"],
            "rejected": data["rejected"],
        }
        results.append(item)

    print(f"  Processed {len(results)} prompts from {benchmark_config['name']}")
    return results


def main():
    """Main function to create PPE_Corr.json dataset."""
    all_data = []

    for benchmark in CORRECTNESS_BENCHMARKS:
        items = process_benchmark(benchmark)
        all_data.extend(items)

    print(f"\nTotal items: {len(all_data)}")

    # Save to JSON file
    output_path = "PPE_Corr.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

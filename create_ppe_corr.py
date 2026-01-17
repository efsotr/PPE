#!/usr/bin/env python3
"""
Script to download and build PPE Correctness dataset for calculating Accuracy metric.

This script downloads the 5 correctness benchmarks from HuggingFace and creates
a unified dataset in the required format for computing the Accuracy metric.
"""

import json
from datasets import load_dataset
from tqdm import tqdm


# Define the 5 correctness benchmarks
CORRECTNESS_BENCHMARKS = {
    "mmlu_pro": "lmarena-ai/PPE-MMLU-Pro-Best-of-K",
    "math": "lmarena-ai/PPE-MATH-Best-of-K",
    "gpqa": "lmarena-ai/PPE-GPQA-Best-of-K",
    "ifeval": "lmarena-ai/PPE-IFEval-Best-of-K",
    "mbpp_plus": "lmarena-ai/PPE-MBPP-Plus-Best-of-K",
}


def process_benchmark(domain: str, dataset_path: str, split: str = "train") -> list:
    """
    Process a single benchmark dataset and extract chosen/rejected responses.
    
    Args:
        domain: The domain name (e.g., "mmlu_pro", "math")
        dataset_path: HuggingFace dataset path
        split: Dataset split to use (default: "train")
    
    Returns:
        List of processed items in the required format
    """
    print(f"Processing {domain} benchmark from {dataset_path}...")
    
    # Load the dataset
    try:
        dataset = load_dataset(dataset_path, split=split)
    except Exception as e:
        print(f"Failed to load dataset {dataset_path}: {e}")
        raise
    
    results = []
    
    # Process each prompt
    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {domain}")):
        prompt = item["prompt"]
        question_id = item["question_id"]
        scores = item["scores"]
        
        # Separate responses into chosen (correct) and rejected (incorrect)
        chosen = []
        rejected = []
        
        # Each benchmark has 32 responses (response_1 to response_32)
        for i in range(32):
            response_key = f"response_{i + 1}"
            response = item[response_key]
            score = scores[i]
            
            # Score 1 means correct (chosen), 0 means incorrect (rejected)
            if score == 1:
                chosen.append(response)
            else:
                rejected.append(response)
        
        # Create the output item
        result_item = {
            "id": f"{domain}_{question_id}",
            "domain": domain,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }
        
        results.append(result_item)
    
    return results


def create_ppe_corr_dataset(output_path: str = "PPE_Corr.json"):
    """
    Download and build the complete PPE Correctness dataset.
    
    Args:
        output_path: Path where the output JSON file will be saved
    """
    all_results = []
    
    # Process each benchmark
    for domain, dataset_path in CORRECTNESS_BENCHMARKS.items():
        try:
            benchmark_results = process_benchmark(domain, dataset_path)
            all_results.extend(benchmark_results)
            print(f"✓ Successfully processed {domain}: {len(benchmark_results)} items")
        except Exception as e:
            print(f"✗ Error processing {domain}: {e}")
            continue
    
    # Save to JSON file
    print(f"\nSaving {len(all_results)} items to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully created {output_path}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total items: {len(all_results)}")
    
    # Count by domain
    domain_counts = {}
    for item in all_results:
        domain = item["domain"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("\nItems per domain:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    
    # Sample statistics
    if all_results:
        sample_item = all_results[0]
        print(f"\nSample item (first from {sample_item['domain']}):")
        print(f"  ID: {sample_item['id']}")
        print(f"  Chosen responses: {len(sample_item['chosen'])}")
        print(f"  Rejected responses: {len(sample_item['rejected'])}")


if __name__ == "__main__":
    create_ppe_corr_dataset()

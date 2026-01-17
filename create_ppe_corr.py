#!/usr/bin/env python3
"""
Script to download and build PPE Correctness dataset for calculating Accuracy metric.

This script downloads the 5 correctness benchmarks from HuggingFace and creates
a unified dataset in the required format for computing the Accuracy metric.

The output format is:
    List[{
        "id": str,           # Unique identifier: "{domain}_{question_id}"
        "domain": str,       # One of: mmlu_pro, math, gpqa, ifeval, mbpp_plus
        "prompt": str,       # The question/prompt text
        "chosen": List[str], # List of correct responses (score=1)
        "rejected": List[str] # List of incorrect responses (score=0)
    }]

Usage:
    python create_ppe_corr.py [--output OUTPUT_PATH] [--split SPLIT]
    
    --output: Output JSON file path (default: PPE_Corr.json)
    --split: Dataset split to use (default: train)
"""

import argparse
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

# Number of responses per question in each benchmark
RESPONSES_PER_QUESTION = 32


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
    except (ConnectionError, OSError) as e:
        error_msg = (
            f"Failed to load dataset '{dataset_path}' for domain '{domain}'.\n"
            f"Network error: {e}\n"
            f"Please check:\n"
            f"  1. Your internet connection is working\n"
            f"  2. You have access to HuggingFace datasets"
        )
        print(error_msg)
        raise
    except ValueError as e:
        error_msg = (
            f"Failed to load dataset '{dataset_path}' for domain '{domain}'.\n"
            f"Invalid dataset configuration: {e}\n"
            f"Please check that the dataset path and split are correct."
        )
        print(error_msg)
        raise
    except Exception as e:
        error_msg = (
            f"Failed to load dataset '{dataset_path}' for domain '{domain}'.\n"
            f"Unexpected error: {e}\n"
            f"Please check:\n"
            f"  1. The dataset path is correct\n"
            f"  2. The split '{split}' exists in the dataset"
        )
        print(error_msg)
        raise
    
    results = []
    
    # Process each prompt
    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {domain}")):
        try:
            prompt = item["prompt"]
            question_id = item["question_id"]
            scores = item["scores"]
        except KeyError as e:
            print(f"Warning: Skipping item {idx} in {domain} - missing field {e}")
            continue
        
        # Validate that we have the expected number of responses
        if len(scores) != RESPONSES_PER_QUESTION:
            print(f"Warning: Item {question_id} in {domain} has {len(scores)} scores, expected {RESPONSES_PER_QUESTION}")
            # Use the actual number of scores available
            num_responses = min(len(scores), RESPONSES_PER_QUESTION)
        else:
            num_responses = RESPONSES_PER_QUESTION
        
        # Separate responses into chosen (correct) and rejected (incorrect)
        chosen = []
        rejected = []
        
        # Process each response up to the number of available scores
        for i in range(num_responses):
            response_key = f"response_{i + 1}"
            
            # Check if response exists
            if response_key not in item:
                print(f"Warning: Missing {response_key} for {question_id} in {domain}")
                continue
            
            # Check if score index is valid
            if i >= len(scores):
                print(f"Warning: Missing score index {i} for {question_id} in {domain}")
                continue
                
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


def create_ppe_corr_dataset(output_path: str = "PPE_Corr.json", split: str = "train"):
    """
    Download and build the complete PPE Correctness dataset.
    
    Args:
        output_path: Path where the output JSON file will be saved
        split: Dataset split to use (default: "train")
    """
    all_results = []
    
    # Process each benchmark
    for domain, dataset_path in CORRECTNESS_BENCHMARKS.items():
        try:
            benchmark_results = process_benchmark(domain, dataset_path, split=split)
            all_results.extend(benchmark_results)
            print(f"✓ Successfully processed {domain}: {len(benchmark_results)} items")
        except (ConnectionError, OSError) as e:
            print(f"✗ Network error processing {domain}: {e}")
            print(f"  Skipping {domain} and continuing with other benchmarks...")
            continue
        except KeyError as e:
            print(f"✗ Data format error in {domain}: Missing expected field {e}")
            print(f"  Skipping {domain} and continuing with other benchmarks...")
            continue
        except Exception as e:
            print(f"✗ Unexpected error processing {domain}: {e}")
            print(f"  Skipping {domain} and continuing with other benchmarks...")
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
    parser = argparse.ArgumentParser(
        description="Download and build PPE Correctness dataset for Accuracy metric calculation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="PPE_Corr.json",
        help="Output JSON file path (default: PPE_Corr.json)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    
    args = parser.parse_args()
    
    create_ppe_corr_dataset(output_path=args.output, split=args.split)

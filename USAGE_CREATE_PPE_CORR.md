# PPE Correctness Dataset Builder

## Overview

The `create_ppe_corr.py` script downloads and builds a unified dataset from the 5 PPE Correctness benchmarks for calculating the Accuracy metric. This dataset is used to evaluate how well reward models and LLM judges can distinguish between correct and incorrect responses.

## Benchmarks Included

The script processes the following 5 correctness benchmarks:

1. **MMLU-Pro** (`lmarena-ai/PPE-MMLU-Pro-Best-of-K`) - General knowledge
2. **MATH** (`lmarena-ai/PPE-MATH-Best-of-K`) - Mathematics
3. **GPQA** (`lmarena-ai/PPE-GPQA-Best-of-K`) - STEM questions
4. **IFEval** (`lmarena-ai/PPE-IFEval-Best-of-K`) - Instruction following
5. **MBPP-Plus** (`lmarena-ai/PPE-MBPP-Plus-Best-of-K`) - Coding

Each benchmark contains 512 prompts with 32 responses each, labeled with correctness scores (1 for correct, 0 for incorrect).

## Usage

### Basic Usage

```bash
python create_ppe_corr.py
```

This will download all 5 benchmarks and create `PPE_Corr.json` in the current directory.

### Advanced Usage

```bash
python create_ppe_corr.py --output custom_output.json --split train
```

**Arguments:**
- `--output`: Output JSON file path (default: `PPE_Corr.json`)
- `--split`: Dataset split to use (default: `train`)

## Output Format

The script generates a JSON file with the following structure:

```json
[
  {
    "id": "mmlu_pro_question_123",
    "domain": "mmlu_pro",
    "prompt": "What is the capital of France?",
    "chosen": [
      "Paris",
      "The capital of France is Paris."
    ],
    "rejected": [
      "London",
      "Berlin"
    ]
  },
  ...
]
```

**Fields:**
- `id` (str): Unique identifier in format `{domain}_{question_id}`
- `domain` (str): One of `mmlu_pro`, `math`, `gpqa`, `ifeval`, `mbpp_plus`
- `prompt` (str): The question or prompt text
- `chosen` (List[str]): All responses marked as correct (score=1)
- `rejected` (List[str]): All responses marked as incorrect (score=0)

**Note:** Responses are kept in order and are not deduplicated, as per requirements.

## Requirements

- Python 3.7+
- `datasets` library
- `tqdm` library
- Internet connection to download from HuggingFace

Install dependencies:
```bash
pip install datasets tqdm
```

Or use the repository's requirements:
```bash
pip install -r requirements.txt
```

## Expected Output

When run successfully, the script will:

1. Download each of the 5 benchmarks from HuggingFace
2. Process 512 prompts from each benchmark
3. Separate responses into correct (chosen) and incorrect (rejected) lists
4. Generate a unified JSON file with approximately 2,560 items (512 × 5)

Example output:
```
Processing mmlu_pro benchmark from lmarena-ai/PPE-MMLU-Pro-Best-of-K...
Processing mmlu_pro: 100%|████████| 512/512 [00:05<00:00, 98.45it/s]
✓ Successfully processed mmlu_pro: 512 items
...
✓ Successfully created PPE_Corr.json

=== Summary ===
Total items: 2560

Items per domain:
  gpqa: 512
  ifeval: 512
  math: 512
  mbpp_plus: 512
  mmlu_pro: 512
```

## Use Cases

This dataset is specifically designed for:

1. **Accuracy Metric Calculation**: Evaluating reward models' and LLM judges' ability to distinguish correct from incorrect responses
2. **Preference Learning**: Training models to prefer correct over incorrect responses
3. **Benchmark Analysis**: Understanding model performance across different domains (knowledge, math, STEM, instructions, coding)

## Relationship to PPE

This script implements the data preparation step for the **Correctness Accuracy** metric described in the PPE (Preference Proxy Evaluations) benchmark. The Accuracy metric measures:

> The accuracy in which the reward model or LLM judge selects the correct answer over the incorrect answer.

For more details on PPE metrics and benchmarks, see the main [README.md](README.md).

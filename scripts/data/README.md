# Data Scripts

Scripts for data preparation and human validation studies.

## Directory Structure

```
scripts/data/
├── download/                           # Dataset download scripts
│   ├── download_sharegpt_data.py       # Download ShareGPT dataset
│   └── download_toucan_data.py         # Download Toucan tool call dataset
│
├── generate_synthetic_datasets.py      # Generate synthetic variation datasets
│
└── human_validation/                   # Human validation study scripts
    │
    ├── likert_rating/                  # Study 1: Pairwise Similarity Rating
    │   ├── generate_dataset.py         # Generate stratified pair samples
    │   ├── export_interface.py         # Export HTML annotation interface
    │   └── analyze_results.py          # Compute correlations (Spearman, Pearson)
    │
    ├── best_match_selection/           # Study 2: Best-Match Selection (RECOMMENDED)
    │   ├── generate_dataset.py         # Find samples where methods disagree
    │   ├── export_interface.py         # Export selection interface
    │   └── analyze_results.py          # Compute win rates, MRR, significance
    │
    └── consistency_ranking/            # Study 3: Consistency Score Validation
        ├── generate_dataset.py         # Generate set comparison pairs
        ├── export_interface.py         # Export ranking interface
        └── analyze_results.py          # Compute accuracy by difficulty
```

## Quick Start

### Download Data

```bash
# Download Toucan tool call dataset
python scripts/data/download/download_toucan_data.py

# Download ShareGPT dataset
python scripts/data/download/download_sharegpt_data.py
```

### Human Validation Study 2: Best-Match Selection (Recommended)

```bash
# Step 1: Generate dataset (finds samples where STED/BERTScore/DeepDiff/TED disagree)
python scripts/data/human_validation/best_match_selection/generate_dataset.py \
    --toucan-data-path toucan_data/toucan_tool_calls_1006.json \
    --results-dir llm_gen_results \
    --output ranking_validation_dataset.json

# Step 2: Export annotation interface
python scripts/data/human_validation/best_match_selection/export_interface.py \
    --input ranking_validation_dataset.json \
    --output-dir best_match_annotation

# Step 3: Annotators open best_match_annotation/index.html in browser

# Step 4: Analyze results
python scripts/data/human_validation/best_match_selection/analyze_results.py \
    --dataset ranking_validation_dataset.json \
    --annotations best_match_annotation/best_match_annotations.csv \
    --output best_match_report.txt
```

### Human Validation Study 1: Likert Rating

```bash
# Generate dataset
python scripts/data/human_validation/likert_rating/generate_dataset.py \
    --toucan-only \
    --output human_validation_dataset.json

# Export interface
python scripts/data/human_validation/likert_rating/export_interface.py \
    --input human_validation_dataset.json \
    --output-dir annotation_interface

# Analyze results
python scripts/data/human_validation/likert_rating/analyze_results.py \
    --dataset human_validation_dataset.json \
    --annotations collected_annotations.csv \
    --output validation_report.txt
```

### Human Validation Study 3: Consistency Ranking

```bash
# Generate dataset
python scripts/data/human_validation/consistency_ranking/generate_dataset.py \
    --llm-results-dir llm_gen_results \
    --output consistency_ranking_dataset.json

# Export interface
python scripts/data/human_validation/consistency_ranking/export_interface.py \
    --input consistency_ranking_dataset.json \
    --output-dir ranking_interface

# Analyze results
python scripts/data/human_validation/consistency_ranking/analyze_results.py \
    --dataset consistency_ranking_dataset.json \
    --annotations ranking_annotations.csv \
    --output ranking_report.txt
```

## Study Comparison

| Study | Task | Metric | Validates |
|-------|------|--------|-----------|
| Likert Rating | Rate pair similarity (1-5) | Spearman correlation | STED similarity scores |
| Best-Match Selection | Choose most similar response | Win rate, MRR | STED's ranking capability |
| Consistency Ranking | Compare set consistency | Accuracy vs random | STED consistency scores |

**Recommendation**: Use **Best-Match Selection** (Study 2) for primary validation - it's more objective and has higher inter-annotator agreement.

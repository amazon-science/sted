# Project Reorganization Plan

**Date**: 2026-01-30
**Status**: COMPLETED

## Current Issues

### 1. Root Directory Clutter
The root directory contains many files that should be organized elsewhere:
- Temporary analysis scripts (`analyze_dataset_distributions.py`, `analyze_nested_json.py`)
- Test files (`test_gemini3.py`, `test_minimax.py`)
- Temporary JSON files (`best_match_report_results.json`, `high_token_samples.json`, `models_to_rerun.json`)
- Large archives (`llm_gen_results.zip`, `sted-internal.zip`, `nemotron-results.tar.gz`)
- LaTeX build artifacts (`sted_theory_icml.aux`, `.log`, `.out`)

### 2. Duplicate/Scattered Results
Results are scattered across multiple locations:
- `results/` - Main results directory
- `consistency_results/` - Redundant results directory
- `research/experiments/` - Experiment-specific results
- `research/ablation_results/` - Ablation study results
- `research/analysis_results/` - Analysis results

### 3. Temporary Experiment Directories
Temporary experiment directories in root:
- `temp_test/` - Should be in `.gitignore` or deleted
- `temperature_experiment/` - Should be in `research/experiments/`

### 4. Documentation Scattered
Documentation files are spread across:
- Root: `README.md`, `LIBRARY_USAGE.md`, `SCRIPTS_REFERENCE.md`, `CONTRIBUTING.md`
- `docs/` - Main documentation
- `research/papers/` - Research notes
- Various `*.md` files in subdirectories

### 5. Data Files Location
Data files are in multiple locations:
- `toucan_data/` - Toucan dataset
- `sharegpt_data/` - ShareGPT dataset
- `llm_gen_results/` - Generated results
- `scripts/data/` - Data processing scripts + data files

---

## Proposed Directory Structure

```
sted-internal/
├── sted/                           # Core library (KEEP AS-IS)
│   ├── __init__.py
│   ├── semantic_json_tree_consistency.py
│   ├── structural_consistency_analyzer.py
│   ├── bedrock_utils.py
│   ├── model_config.py
│   ├── utils.py
│   └── ...
│
├── scripts/                        # All executable scripts
│   ├── analysis/                   # Analysis scripts
│   ├── data/                       # Data processing scripts
│   ├── eval/                       # Evaluation scripts
│   ├── experiments/                # Experiment runner scripts
│   └── visualization/              # Visualization scripts
│
├── data/                           # ALL data files (NEW - consolidated)
│   ├── toucan/                     # Toucan benchmark data
│   ├── sharegpt/                   # ShareGPT benchmark data
│   └── human_validation/           # Human validation datasets
│
├── results/                        # ALL results (consolidated)
│   ├── toucan/                     # Toucan experiment results
│   ├── sharegpt/                   # ShareGPT experiment results
│   ├── ablation/                   # Ablation study results
│   └── human_validation/           # Human validation results
│
├── llm_gen_results/                # Raw LLM generation outputs (KEEP)
│   ├── toucan/
│   └── sharegpt/
│
├── research/                       # Research artifacts
│   ├── experiments/                # Experiment code and intermediate results
│   ├── papers/                     # Research notes and drafts
│   └── notebooks/                  # Jupyter notebooks
│
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   ├── papers/                     # Published papers
│   │   ├── ICML_paper/
│   │   └── KDD_paper/
│   ├── guides/                     # User guides
│   │   ├── EC2_EXPERIMENT_SETUP.md
│   │   ├── LIBRARY_USAGE.md
│   │   └── SCRIPTS_REFERENCE.md
│   └── analysis/                   # Analysis reports
│       ├── EMBEDDING_MODEL_ABLATION.md
│       ├── HUMAN_VALIDATION_STUDY.md
│       ├── MODEL_CHARACTERISTICS_ANALYSIS.md
│       └── prompt_inconsistency_analysis.md
│
├── tests/                          # Unit and integration tests
│
├── figures/                        # Generated figures for papers
│
├── examples/                       # Example usage scripts
│
├── benchmarks/                     # Benchmark definitions
│
├── .archive/                       # Archived/deprecated files (NEW)
│
├── README.md                       # Main readme
├── pyproject.toml                  # Python project config
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── NOTICE
```

---

## Action Items

### Phase 1: Clean Root Directory

| File | Action | Destination |
|------|--------|-------------|
| `analyze_dataset_distributions.py` | Move | `scripts/analysis/` |
| `analyze_nested_json.py` | Move | `scripts/analysis/` |
| `test_gemini3.py` | Move | `tests/integration/` or Delete |
| `test_minimax.py` | Move | `tests/integration/` or Delete |
| `best_match_report_results.json` | Move | `results/human_validation/` |
| `best_match_report.txt` | Move | `results/human_validation/` |
| `high_token_samples.json` | Move | `.archive/` or Delete |
| `models_to_rerun.json` | Move | `.archive/` or Delete |
| `ranking_validation_dataset.json` | Move | `data/human_validation/` |
| `llm_gen_results.zip` | Delete | (can regenerate from S3) |
| `sted-internal.zip` | Delete | (project archive, not needed) |
| `nemotron-results.tar.gz` | Move | `.archive/` |
| `sted_theory_icml.*` (aux,log,out,pdf) | Delete | (LaTeX build artifacts in root) |
| `qwen235b_experiment.log` | Delete | (temporary log) |
| `texput.log` | Delete | (LaTeX temp file) |
| `human_validation_interface.zip` | Delete | (has unzipped version) |
| `DATASET_EVALUATION_REPORT.md` | Move | `docs/analysis/` |
| `LIBRARY_USAGE.md` | Move | `docs/guides/` |
| `SCRIPTS_REFERENCE.md` | Move | `docs/guides/` |

### Phase 2: Consolidate Data Directories

| Current Location | Action | New Location |
|------------------|--------|--------------|
| `toucan_data/` | Move | `data/toucan/` |
| `sharegpt_data/` | Move | `data/sharegpt/` |
| `scripts/data/human_validation/*/data/` | Move | `data/human_validation/` |

### Phase 3: Consolidate Results

| Current Location | Action | New Location |
|------------------|--------|--------------|
| `consistency_results/` | Merge | `results/` then delete |
| `research/ablation_results/` | Move | `results/ablation/` |
| `research/analysis_results/` | Move | `results/analysis/` |

### Phase 4: Clean Temporary Directories

| Directory | Action |
|-----------|--------|
| `temp_test/` | Delete (add to .gitignore) |
| `temperature_experiment/` | Move to `research/experiments/` or delete |
| `mcp_dev/` | Move to `.archive/` or delete if unused |

### Phase 5: Reorganize Documentation

| Current Location | Action | New Location |
|------------------|--------|--------------|
| `docs/ICML_paper/` | Keep | `docs/papers/ICML_paper/` |
| `docs/KDD_paper/` | Keep | `docs/papers/KDD_paper/` |
| `docs/EC2_EXPERIMENT_SETUP.md` | Move | `docs/guides/` |
| `docs/STED_*.md/pdf/html` | Move | `docs/guides/` |
| `research/papers/*.md` | Review | Keep research notes separate |

---

## Files to Add to .gitignore

```gitignore
# Temporary directories
temp_test/
temperature_experiment/

# Large archives (store in S3 instead)
*.zip
*.tar.gz

# LaTeX build artifacts (keep only in docs/papers/)
*.aux
*.log
*.out
*.bbl
*.blg
*.synctex.gz

# OS files
.DS_Store
Thumbs.db

# IDE
.idea/
.vscode/

# Python
__pycache__/
*.pyc
.pytest_cache/

# Environment
.env
.venv/
```

---

## Migration Script

Once approved, I can create a migration script to:
1. Create new directory structure
2. Move files to appropriate locations
3. Update imports in Python files if needed
4. Update documentation references
5. Clean up empty directories

---

## Completed Actions (2026-01-30)

### Files Deleted
- `llm_gen_results.zip`, `sted-internal.zip`, `human_validation_interface.zip` (large archives)
- `sted_theory_icml.aux`, `.log`, `.out`, `.pdf` (LaTeX artifacts from root)
- `qwen235b_experiment.log`, `texput.log` (temp logs)
- `temp_test/` directory

### Files Moved to Scripts
- `analyze_dataset_distributions.py` → `scripts/analysis/`
- `analyze_nested_json.py` → `scripts/analysis/`
- `test_gemini3.py`, `test_minimax.py` → `tests/integration/`

### Data Consolidated
- `toucan_data/` → `data/toucan/`
- `sharegpt_data/` → `data/sharegpt/`
- `human_evaluation/` → `data/human_validation/`

### Results Consolidated
- `consistency_results/` → `.archive/consistency_results_old` (older, smaller version)

### Documentation Reorganized
- `docs/ICML_paper/`, `docs/KDD_paper/` → `docs/papers/`
- `EC2_EXPERIMENT_SETUP.md`, `STED_*.md` → `docs/guides/`
- Analysis reports → `docs/analysis/`

### Archived (`.archive/`)
- `nemotron-results.tar.gz`
- `mcp_dev/`
- `temperature_experiment/`
- `research/ablation_results/`, `research/analysis_results/`
- `research/theoretical_validation/`, `research/triangle_inequality/`
- `high_token_samples.json`, `models_to_rerun.json`

### .gitignore Updated
- Added `.archive/`, `temp_test/`, `temperature_experiment/`
- Added `*.zip`, `*.tar.gz`
- Added LaTeX artifacts

---

## Notes

- Keep `llm_gen_results/` at root level (large, frequently accessed)
- Keep `results/` at root level (primary output location)
- The `sted/` package should remain unchanged (core library)
- Archived files in `.archive/` can be deleted after verification

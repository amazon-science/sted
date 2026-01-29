# Embedding Model Ablation Study

**Date**: 2025-01-28
**Purpose**: Validate that STED consistency scores are robust to the choice of embedding model
**Status**: Complete

---

## Executive Summary

We conducted an ablation study comparing 9 embedding models across 200 Toucan samples. Results show **extremely high correlation (r > 0.995)** between all model pairs, confirming that STED scores are robust to embedding model choice.

**Key Finding**: The choice of embedding model has minimal impact on STED consistency scores (< 1% variance in mean scores).

---

## Experiment Configuration

### Dataset
- **Source**: Toucan benchmark (tool-calling evaluation)
- **Samples**: 200 randomly sampled (seed=42)
- **Responses per sample**: Up to 10 LLM-generated outputs

### Embedding Models Tested

| # | Model | Type | Dimension | Source |
|---|-------|------|-----------|--------|
| 1 | all-MiniLM-L6-v2 | Sentence-Transformer | 384 | Local |
| 2 | all-mpnet-base-v2 | Sentence-Transformer | 768 | Local |
| 3 | BAAI/bge-base-en-v1.5 | Sentence-Transformer | 768 | Local |
| 4 | BAAI/bge-large-en-v1.5 | Sentence-Transformer | 1024 | Local |
| 5 | intfloat/e5-base-v2 | Sentence-Transformer | 768 | Local |
| 6 | intfloat/e5-large-v2 | Sentence-Transformer | 1024 | Local |
| 7 | amazon.titan-embed-text-v2:0 | AWS Bedrock | 256 | API |
| 8 | amazon.titan-embed-text-v2:0 | AWS Bedrock | 512 | API |
| 9 | amazon.titan-embed-text-v2:0 | AWS Bedrock | 1024 | API |

### STED Parameters
- **Variation type**: combined (structural + content)
- **Consistency metric**: c_mean

---

## Results

### Spearman Correlation Matrix (Rank Correlation)

All pairwise Spearman correlations are > 0.995, indicating near-perfect rank agreement.

|                    | MiniLM | mpnet | bge-base | bge-large | e5-base | e5-large | Titan-256 | Titan-512 | Titan-1024 |
|--------------------|--------|-------|----------|-----------|---------|----------|-----------|-----------|------------|
| MiniLM             | 1.0000 | 0.9999| 0.9979   | 0.9979    | 0.9976  | 0.9976   | 0.9978    | 0.9978    | 0.9978     |
| mpnet              | 0.9999 | 1.0000| 0.9981   | 0.9981    | 0.9978  | 0.9978   | 0.9978    | 0.9978    | 0.9978     |
| bge-base           | 0.9979 | 0.9981| 1.0000   | 0.9999    | 0.9998  | 0.9998   | 0.9956    | 0.9956    | 0.9956     |
| bge-large          | 0.9979 | 0.9981| 0.9999   | 1.0000    | 0.9998  | 0.9998   | 0.9956    | 0.9956    | 0.9956     |
| e5-base            | 0.9976 | 0.9978| 0.9998   | 0.9998    | 1.0000  | 0.9999   | 0.9952    | 0.9952    | 0.9952     |
| e5-large           | 0.9976 | 0.9978| 0.9998   | 0.9998    | 0.9999  | 1.0000   | 0.9952    | 0.9952    | 0.9952     |
| Titan-256          | 0.9978 | 0.9978| 0.9956   | 0.9956    | 0.9952  | 0.9952   | 1.0000    | 1.0000    | 1.0000     |
| Titan-512          | 0.9978 | 0.9978| 0.9956   | 0.9956    | 0.9952  | 0.9952   | 1.0000    | 1.0000    | 1.0000     |
| Titan-1024         | 0.9978 | 0.9978| 0.9956   | 0.9956    | 0.9952  | 0.9952   | 1.0000    | 1.0000    | 1.0000     |

**Observations**:
- Minimum correlation: 0.9952 (E5 vs Titan)
- Maximum correlation: 1.0000 (Titan variants are identical)
- Same-family models (e.g., BGE base/large) show r > 0.999

### Pearson Correlation Matrix (Linear Correlation)

|                    | MiniLM | mpnet | bge-base | bge-large | e5-base | e5-large | Titan-256 | Titan-512 | Titan-1024 |
|--------------------|--------|-------|----------|-----------|---------|----------|-----------|-----------|------------|
| MiniLM             | 1.0000 | 1.0000| 0.9998   | 0.9998    | 0.9995  | 0.9995   | 0.9999    | 0.9999    | 0.9999     |
| mpnet              | 1.0000 | 1.0000| 0.9998   | 0.9998    | 0.9995  | 0.9995   | 0.9999    | 0.9999    | 0.9998     |
| bge-base           | 0.9998 | 0.9998| 1.0000   | 1.0000    | 0.9999  | 0.9999   | 0.9998    | 0.9997    | 0.9996     |
| bge-large          | 0.9998 | 0.9998| 1.0000   | 1.0000    | 0.9999  | 0.9999   | 0.9998    | 0.9997    | 0.9996     |
| e5-base            | 0.9995 | 0.9995| 0.9999   | 0.9999    | 1.0000  | 1.0000   | 0.9995    | 0.9993    | 0.9993     |
| e5-large           | 0.9995 | 0.9995| 0.9999   | 0.9999    | 1.0000  | 1.0000   | 0.9995    | 0.9993    | 0.9993     |
| Titan-256          | 0.9999 | 0.9999| 0.9998   | 0.9998    | 0.9995  | 0.9995   | 1.0000    | 1.0000    | 1.0000     |
| Titan-512          | 0.9999 | 0.9999| 0.9997   | 0.9997    | 0.9993  | 0.9993   | 1.0000    | 1.0000    | 1.0000     |
| Titan-1024         | 0.9999 | 0.9998| 0.9996   | 0.9996    | 0.9993  | 0.9993   | 1.0000    | 1.0000    | 1.0000     |

### Summary Statistics (c_mean)

| Model | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| all-MiniLM-L6-v2 | 0.5886 | 0.3014 | 0.0000 | 1.0000 |
| all-mpnet-base-v2 | 0.5887 | 0.3016 | 0.0000 | 1.0000 |
| BAAI/bge-base-en-v1.5 | 0.5911 | 0.3015 | 0.0000 | 1.0000 |
| BAAI/bge-large-en-v1.5 | 0.5911 | 0.3015 | 0.0000 | 1.0000 |
| intfloat/e5-base-v2 | 0.5926 | 0.3014 | 0.0000 | 1.0000 |
| intfloat/e5-large-v2 | 0.5928 | 0.3014 | 0.0000 | 1.0000 |
| Titan-256 | 0.5881 | 0.3015 | 0.0000 | 1.0000 |
| Titan-512 | 0.5874 | 0.3016 | 0.0000 | 1.0000 |
| Titan-1024 | 0.5873 | 0.3016 | 0.0000 | 1.0000 |

**Key Statistics**:
- Mean c_mean range: 0.5873 - 0.5928 (variance < 1%)
- Standard deviation: ~0.301 (consistent across all models)
- All models span full range [0, 1]

---

## Analysis

### Why Such High Correlation?

1. **STED's structural component dominates**: The tree structure similarity contributes significantly to the final score, independent of embedding choice.

2. **Semantic similarity is relative**: Embeddings capture relative semantic distances well, even if absolute representations differ.

3. **Modern embeddings are well-calibrated**: All tested models produce normalized, high-quality embeddings trained on similar objectives.

### Implications for STED

1. **Robustness validated**: Users can choose any reasonable embedding model without affecting results.

2. **Cost flexibility**: Can use free local models (sentence-transformers) instead of paid APIs (Titan) with no quality loss.

3. **Reproducibility**: Results are reproducible across different embedding backends.

---

## Reproduction

### Script
```bash
python scripts/eval/embedding_model_ablation.py \
    --results-dir llm_gen_results/toucan \
    --num-samples 200 \
    --output results/embedding_ablation/ablation_9models.json \
    --region us-east-1 \
    --embedding-models \
        all-MiniLM-L6-v2 \
        all-mpnet-base-v2 \
        BAAI/bge-base-en-v1.5 \
        BAAI/bge-large-en-v1.5 \
        intfloat/e5-base-v2 \
        intfloat/e5-large-v2 \
        amazon.titan-embed-text-v2:0:256 \
        amazon.titan-embed-text-v2:0:512 \
        amazon.titan-embed-text-v2:0:1024
```

### Requirements
- Python 3.10+
- sentence-transformers
- boto3 (for Titan models)
- scipy, numpy, tqdm

### Execution Environment
- EC2 Instance: i-04525d4919abbd4ff (g5.16xlarge)
- Region: us-east-1
- Runtime: ~10 minutes for 9 models x 200 samples

---

## Files

| File | Location | Description |
|------|----------|-------------|
| Results JSON | `results/embedding_ablation/ablation_9models.json` | Full results with per-sample scores |
| Execution Log | `results/embedding_ablation/ablation_9models.log` | Detailed execution log |
| Script | `scripts/eval/embedding_model_ablation.py` | Ablation study script |

### S3 Backup
```
s3://sted-experiment-822507008821/embedding-model-ablation/
├── ablation_9models.json      (318.6 KiB)
├── ablation_9models.log       (453.6 KiB)
└── embedding_model_ablation.py (11.5 KiB)
```

---

## Conclusion

The embedding model ablation study confirms that **STED scores are highly robust to embedding model choice**. With Spearman correlations > 0.995 across all 9 models tested and < 1% variance in mean scores, we can confidently state that:

1. The "minimal impact" claim in the paper is validated
2. Any modern embedding model can be used without affecting STED's evaluation quality
3. Users can choose based on cost/latency preferences rather than accuracy concerns

**Recommendation**: Use `all-MiniLM-L6-v2` as the default for best speed/quality tradeoff, or any sentence-transformer model for reproducibility without API dependencies.

---

*Last updated: 2025-01-28*

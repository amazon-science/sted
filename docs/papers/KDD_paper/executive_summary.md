# Executive Summary: LLM Structured Output Consistency Study

**Paper**: "What Makes LLM Structured Outputs Inconsistent?"
**Venue**: KDD 2026 (Top-tier Data Mining Conference)
**Status**: Ready for submission (Deadline: Feb 8, 2026)

---

## The Problem

When businesses use LLMs to generate structured data (API calls, JSON responses, tool invocations), the same request can produce different outputs each time. This inconsistency causes:
- **System failures** from unexpected output formats
- **Debugging costs** from unpredictable behavior
- **User experience issues** from inconsistent results

**No systematic study existed** to explain *why* this happens or *how* to prevent it.

---

## What We Did

We conducted the **largest study of LLM structured output consistency**:
- **18 models** from 10 providers (Claude, GPT, Llama, Gemini, etc.)
- **2.1 million+ outputs** analyzed
- **42 factors** tested for impact on consistency
- **225,000 evaluation instances** across different configurations

---

## Key Findings

### 1. Schema Design is the #1 Factor
**Complex output schemas cause 2-3x more inconsistency than simple ones.**

| Schema Type | Consistency Score |
|-------------|------------------|
| Simple (flat structure) | 75% |
| Complex (nested objects) | 52% |

*Implication: Simplify your JSON schemas before optimizing anything else.*

### 2. Temperature Settings Must Match Complexity
**Complex schemas degrade 2.5x faster as temperature increases.**

| | Low Temp | High Temp | Degradation |
|---|---------|-----------|-------------|
| Simple schema | 75% | 71% | -6% |
| Complex schema | 61% | 52% | -15% |

*Implication: Use lower temperatures (0.0-0.2) for complex outputs.*

### 3. Tool Naming Matters
**Similar tool names cause selection confusion.**

When tools share prefixes (e.g., `get_user`, `get_user_details`, `get_user_history`), models inconsistently choose between them.

*Implication: Use distinct, unambiguous tool names.*

### 4. Model Selection Impact
**Claude models show highest consistency (72%), followed by GPT (65%) and open-source (58%).**

However, proper configuration can close most of this gap.

---

## Business Impact

### Before (Without Guidelines)
- Average consistency: **58%**
- High variance across deployments
- Trial-and-error optimization

### After (With Our Guidelines)
- Average consistency: **Up to 84%** (+45% improvement)
- Predictable behavior from day one
- Data-driven configuration decisions

---

## Actionable Recommendations

| Priority | Action | Expected Improvement |
|----------|--------|---------------------|
| **1** | Flatten nested schemas to depth ≤2 | +15-20% |
| **2** | Reduce parameters per tool to ≤10 | +10-15% |
| **3** | Set temperature based on complexity | +10-25% |
| **4** | Use distinct tool names | +5-10% |

**Total potential improvement: Up to 45%**

---

## Why This Matters for KDD

- **First comprehensive factor analysis** of LLM structured outputs
- **Practical guidelines** backed by 2M+ data points
- **Immediately applicable** to any LLM deployment
- **Novel findings** (temperature-complexity interaction, tool naming effects)

---

## One-Line Summary

> **We discovered that schema complexity—not the query or model—is the primary driver of LLM output inconsistency, and provide guidelines that can improve consistency by up to 45%.**

---

*Contact: [Author] | Full paper: docs/KDD_paper/kdd2025_submission.pdf*

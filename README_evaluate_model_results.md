# Multi-Model Results Evaluation Script

This document provides comprehensive instructions for using the `evaluate_model_results.py` script to evaluate and compare existing model results from different Claude variants.

## 🚀 **Script Overview**

The `evaluate_model_results.py` script is specifically designed for evaluating existing model results and creating comprehensive comparisons focusing on the three primary consistency metrics:

1. **Consistency Coefficient** - Combines accuracy and stability
2. **Normalized CV** - Scale-independent variability measure  
3. **Stability Score** - Intuitive 0-1 stability measure

## 🎯 **Key Features**

### **Core Capabilities:**
- **Automatic Model Discovery** - Finds all `generations-*` folders
- **Temperature Extraction** - Automatically extracts temperatures from folder names
- **Primary Metrics Focus** - Concentrates on the 3 key consistency metrics
- **Cross-Model Comparison** - Comprehensive analysis across all models
- **Rich Visualizations** - Multiple comparison plots and overview charts
- **Detailed Reporting** - Both human-readable and JSON outputs

### **Advanced Features:**
- **Robust Error Handling** - Continues processing even if some files fail
- **Progress Tracking** - Shows progress bars and status updates
- **Outlier Detection** - Optional outlier removal for cleaner correlations
- **Statistical Analysis** - P-values and correlation significance testing
- **Best Model Identification** - Automatically finds top performers

## 📋 **Usage Instructions**

### **Basic Usage:**
```bash
python evaluate_model_results.py --results-dir ./temperature_experiment --output-dir ./model_comparison_results
```

### **Advanced Usage:**
```bash
python evaluate_model_results.py \
    --results-dir ./temperature_experiment \
    --output-dir ./model_comparison_results \
    --methods ted bertscore deepdiff \
    --embedding-model amazon.titan-embed-text-v2:0
```

### **Command Line Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--results-dir` | ✅ Yes | - | Directory containing model result folders |
| `--output-dir` | ❌ No | `./model_comparison_results` | Directory to save comparison results |
| `--methods` | ❌ No | `["ted", "bertscore", "deepdiff"]` | Similarity methods to use |
| `--embedding-model` | ❌ No | `amazon.titan-embed-text-v2:0` | Embedding model ID |
| `--remove-outliers` | ❌ No | `False` | Remove outliers before correlation analysis |

## 📁 **Expected Input Structure**

Your input directory should contain model result folders with the following structure:

```
temperature_experiment/
├── generations-claude-3-5-sonnet-v2/
│   ├── llm_gen_results_temp_0_00_20250108_123456/
│   │   └── all_results.json
│   ├── llm_gen_results_temp_0_05_20250108_123456/
│   │   └── all_results.json
│   ├── llm_gen_results_temp_0_10_20250108_123456/
│   │   └── all_results.json
│   └── ...
├── generations-claude-3-haiku/
│   ├── llm_gen_results_temp_0_00_20250108_123456/
│   │   └── all_results.json
│   ├── llm_gen_results_temp_0_05_20250108_123456/
│   │   └── all_results.json
│   └── ...
├── generations-claude-3-5-sonnet-20240620/
│   ├── llm_gen_results_temp_0_00_20250108_123456/
│   │   └── all_results.json
│   └── ...
└── ...
```

### **Key Requirements:**
- Folders must start with `generations-`
- Each temperature folder must contain `all_results.json`
- Temperature must be extractable from folder name (e.g., `temp_0_50` → 0.50)

## 📊 **Output Files Generated**

### **Main Analysis Files:**
```
model_comparison_YYYYMMDD_HHMMSS/
├── model_comparison_summary.txt          # 📄 Main human-readable report
├── model_comparison_summary.json         # 📊 Structured data for analysis
```

### **Visualizations:**
```
├── model_comparison_consistency_coefficient.png  # 🎯 Primary Metric #1
├── model_comparison_std_normalized_cv.png        # 📈 Primary Metric #2
├── model_comparison_stability_score.png          # 🎯 Primary Metric #3
├── comprehensive_model_comparison.png            # 🔍 All metrics overview
```

### **Per-Temperature Analysis:**
```
├── generations-claude-3-5-sonnet-v2/
│   ├── temp_0.00_analysis/
│   │   ├── primary_metrics_comparison.txt
│   │   └── primary_metrics_comparison.json
│   ├── temp_0.05_analysis/
│   │   ├── primary_metrics_comparison.txt
│   │   └── primary_metrics_comparison.json
│   └── ...
```

## 🎯 **Primary Metrics Explained**

### **1. Consistency Coefficient (Higher = Better)**
- **Formula**: `mean * (1 - min(CV^1.5, 1.0))`
- **Purpose**: Combines accuracy and stability in one metric
- **Interpretation**: Higher values indicate better overall consistency
- **Temperature Correlation**: Should be negative (higher temp → lower consistency)

### **2. Normalized CV (Lower = More Stable)**
- **Formula**: `mean(std / mean)` across samples
- **Purpose**: Scale-independent measure of variability
- **Interpretation**: Lower values indicate more stable outputs
- **Temperature Correlation**: Should be positive (higher temp → more variability)

### **3. Stability Score (Higher = More Stable)**
- **Formula**: `1.0 / (1.0 + mean_of_stds)`
- **Purpose**: Intuitive 0-1 scale stability measure
- **Interpretation**: Higher values indicate more stable outputs
- **Temperature Correlation**: Should be negative (higher temp → less stability)

## 📈 **Understanding the Analysis**

### **Correlation Analysis:**
The script analyzes how each primary metric correlates with temperature:

- **Strong Correlation**: |r| > 0.7, p < 0.05
- **Moderate Correlation**: 0.3 < |r| < 0.7, p < 0.05
- **Weak Correlation**: |r| < 0.3 or p > 0.05

### **Best Model Identification:**
For each metric, the script identifies the model with the strongest temperature correlation:

- **Consistency Coefficient**: Most negative correlation (most temperature-sensitive)
- **Normalized CV**: Most positive correlation (most temperature-sensitive)
- **Stability Score**: Most negative correlation (most temperature-sensitive)

### **Overall Best Model:**
Determined by averaging the absolute correlation strengths across all metrics and methods.

## 💡 **Example Output Summary**

When you run the script, you'll get insights like:

```
🤖 MULTI-MODEL COMPARISON SUMMARY
============================================================

Analysis Date: 2025-01-08 14:30:25
Models Compared: claude-3-5-sonnet-v2, claude-3-haiku, claude-3-5-sonnet-20240620
Methods Used: TED, BERTSCORE, DEEPDIFF
Temperature Range: 0.00 - 1.00
Total Data Points: 63 temperature-model combinations

🎯 PRIMARY METRICS FOCUS:
• Consistency Coefficient: Combines accuracy and stability (Higher = Better)
• Normalized CV: Scale-independent variability (Lower = More Stable)
• Stability Score: Intuitive 0-1 stability measure (Higher = More Stable)

================================================================================
📊 TED METHOD - TEMPERATURE CORRELATIONS
================================================================================
Model                     Consistency Coeff    Normalized CV        Stability Score     
                         (r, p-value)         (r, p-value)         (r, p-value)        
------------------------------------------------------------------------------------------
3.5 sonnet v2            (-0.847, 0.001)      (+0.923, 0.000)      (-0.756, 0.002)     
3.5 haiku                (-0.723, 0.003)      (+0.891, 0.000)      (-0.689, 0.005)     
3.5 sonnet 20240620      (-0.692, 0.006)      (+0.834, 0.001)      (-0.645, 0.008)     

================================================================================
🏆 BEST PERFORMING MODELS BY METRIC
================================================================================

TED Method:
  • Consistency Coefficient: 3.5 sonnet v2 (r = -0.847)
  • Normalized CV: 3.5 sonnet v2 (r = +0.923)
  • Stability Score: 3.5 sonnet v2 (r = -0.756)

================================================================================
💡 RECOMMENDATIONS
================================================================================
🥇 Overall Best Model: 3.5 sonnet v2
   → Most consistent temperature-stability correlations across metrics

📈 Temperature Recommendations:
• For maximum consistency: Use temperature ≤ 0.3
• For balanced creativity/consistency: Use temperature 0.5-0.7
• Monitor Consistency Coefficient as primary metric
```

## 🔧 **Script Advantages**

### **vs. Modifying Existing Scripts:**
- ✅ **Cleaner Code**: Single-purpose, focused functionality
- ✅ **Better Maintainability**: Easier to understand and modify
- ✅ **Robust Error Handling**: Continues processing despite individual failures
- ✅ **Rich Output**: More comprehensive analysis and reporting
- ✅ **Flexible Input**: Works with any folder structure matching the pattern

### **Key Improvements:**
- **Automatic Model Detection**: No need to specify model names
- **Smart Temperature Extraction**: Handles various naming conventions
- **Comprehensive Reporting**: Both technical and executive summaries
- **Publication-Ready Plots**: Professional visualizations for papers
- **JSON Export**: Easy integration with other analysis tools

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **1. "No model result folders found"**
- **Cause**: Folders don't start with `generations-`
- **Solution**: Ensure folder names follow the pattern `generations-{model-name}`

#### **2. "Could not extract temperature from path"**
- **Cause**: Temperature not in expected format in folder name
- **Solution**: Ensure folder names contain `temp_X_YY` pattern (e.g., `temp_0_50`)

#### **3. "No result files found for model"**
- **Cause**: Missing `all_results.json` files
- **Solution**: Ensure each temperature folder contains `all_results.json`

#### **4. "Error processing result file"**
- **Cause**: Corrupted or invalid JSON files
- **Solution**: Check JSON file validity; script will continue with other files

### **Performance Tips:**

#### **For Large Datasets:**
- Use `--remove-outliers` to improve correlation quality
- Consider processing subsets of temperatures if memory is limited
- Monitor disk space for output files and visualizations

#### **For Better Visualizations:**
- Ensure model names are descriptive but not too long
- Use consistent temperature ranges across models
- Check that all models have similar temperature coverage

## 📚 **Integration with Other Tools**

### **Using JSON Output:**
```python
import json

# Load results for further analysis
with open('model_comparison_summary.json', 'r') as f:
    results = json.load(f)

# Access correlation data
correlations = results['correlations_by_model']
for model, methods in correlations.items():
    for method, metrics in methods.items():
        cc_correlation = metrics['consistency_coefficient']['pearson']['r']
        print(f"{model} {method}: CC correlation = {cc_correlation:.3f}")
```

### **Extending the Analysis:**
The script is designed to be easily extensible:

- **Add New Metrics**: Modify `extract_temperature_metrics()` function
- **Custom Visualizations**: Extend `create_model_comparison_plots()` function
- **Different Models**: Works automatically with any `generations-*` folders
- **Additional Methods**: Add to `--methods` argument

## 🎯 **Research Applications**

### **For Academic Papers:**
- **Model Comparison Studies**: Compare different LLM variants
- **Temperature Analysis**: Understand temperature effects on consistency
- **Evaluation Method Validation**: Compare TED vs. BERTScore vs. DeepDiff
- **Consistency Metrics**: Validate new consistency measurement approaches

### **For Production Systems:**
- **Model Selection**: Choose optimal model for consistency requirements
- **Temperature Tuning**: Find optimal temperature settings
- **Quality Monitoring**: Track consistency metrics over time
- **A/B Testing**: Compare model performance in production

## 📝 **Citation**

If you use this script in your research, please cite:

```
@software{evaluate_model_results,
  title={Multi-Model Consistency Evaluation Script},
  author={[Your Name]},
  year={2025},
  url={[Your Repository URL]}
}
```

## 🤝 **Contributing**

To contribute improvements:

1. **Fork the repository**
2. **Create a feature branch**
3. **Add your improvements**
4. **Test with different model configurations**
5. **Submit a pull request**

### **Areas for Contribution:**
- Additional visualization types
- New consistency metrics
- Performance optimizations
- Support for other model formats
- Enhanced error handling

---

**Last Updated**: January 8, 2025  
**Version**: 1.0  
**Compatibility**: Python 3.8+, requires pandas, matplotlib, seaborn, scipy, tqdm
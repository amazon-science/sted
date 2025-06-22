#!/usr/bin/env python3

# This script fixes the issues in consistency_eval.py

import re

# Read the original file
with open('consistency_eval.py', 'r') as f:
    content = f.read()

# Fix 1: Add safety check for 'consistency_analysis' key in calculate_prompt_consistency_async
pattern1 = r"if field_metrics\[self\.primary_field\]\['consistency_analysis'\]:"
replacement1 = "if self.primary_field in field_metrics and 'consistency_analysis' in field_metrics[self.primary_field] and field_metrics[self.primary_field]['consistency_analysis']:"
content = re.sub(pattern1, replacement1, content)

# Fix 2: Add safety check for 'consistency_analysis' key in calculate_prompt_consistency
pattern2 = r"if field_metrics\[self\.primary_field\]\['consistency_analysis'\]:"
replacement2 = "if self.primary_field in field_metrics and 'consistency_analysis' in field_metrics[self.primary_field] and field_metrics[self.primary_field]['consistency_analysis']:"
content = content.replace(pattern2, replacement2, 1)  # Replace only the first occurrence after the first replacement

# Fix 3: Fix floating point comparison in test_calculate_field_consistency_nested
# This is in the test file, not consistency_eval.py, so we'll handle it separately

# Write the fixed content back
with open('consistency_eval_fixed.py', 'w') as f:
    f.write(content)

print("Fixed version written to consistency_eval_fixed.py")

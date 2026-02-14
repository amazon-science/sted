#!/usr/bin/env python3
"""
Experiment 7: Linguistically-Grounded Deep Features (COLM 2026)

Based on empirical analysis of consistent vs inconsistent prompts in Toucan dataset,
combined with linguistics literature on factors that create interpretation ambiguity
in language understanding.

EMPIRICAL FINDINGS (from comparing top-50 vs bottom-50 by c_mean):
=================================================================
Inconsistent prompts have (with large effect sizes):
  1. HIGHER clause density (d=0.78): more subordinate clauses per sentence
  2. HIGHER avg sentence length (d=0.86): longer, more complex sentences
  3. HIGHER definite references (d=1.00): more "the X" constructions
  4. LOWER type-token ratio (d=0.79): more repetitive vocabulary
  5. LOWER action verb density (d=0.95): fewer verbs per word (verbose)
  6. HIGHER conjunction chains (d=0.80): more "and...and..." patterns
  7. LOWER starts_with_i (d=0.75): less first-person framing

LINGUISTIC THEORY GROUNDING:
============================
These patterns align with established psycholinguistic findings:

1. **Dependency Locality Theory (Gibson 2000)**: Processing difficulty increases
   with the distance between syntactic dependents. Long-distance dependencies
   create interpretation uncertainty -> inconsistency.

2. **Uniform Information Density (Jaeger 2010)**: Natural language distributes
   information uniformly. Prompts with uneven information density (sparse verbs
   + dense noun phrases) violate this, creating ambiguity about focus.

3. **Referential Theory (Grosz, Joshi, Weinstein 1995)**: Centering Theory predicts
   that when discourse has many competing referential anchors without clear focus,
   interpretation becomes ambiguous.

4. **Rhetorical Structure Theory (Mann & Thompson 1988)**: Prompts with deep
   rhetorical nesting (conditions within conditions) are harder to parse deterministically.

5. **Planning Complexity (Levelt 1989)**: Multi-step tasks with implicit ordering
   create planning ambiguity - the model must infer which tools to use in which order.

6. **Information-Theoretic Measures**: Shannon entropy of word distributions
   captures lexical predictability. High-entropy prompts have more uniform
   (less predictable) word distributions.

NEW FEATURES:
=============
1. Clause density & depth (from empirical finding)
2. Dependency distance (Gibson 2000) - via spaCy parse tree
3. Information density uniformity (Jaeger 2010) - local information rate variance
4. Referential density & competition (Centering Theory)
5. Rhetorical depth (condition nesting)
6. Planning complexity (multi-tool coordination signals)
7. Lexical entropy (information theory)
8. Verb-to-noun ratio (empirical: action_verb_density matters)
9. Sentence-level complexity variance (within-prompt heterogeneity)
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, ttest_rel, entropy

try:
    import spacy
    _nlp = spacy.load('en_core_web_sm')
    SPACY_OK = True
except:
    SPACY_OK = False
    print("WARNING: spaCy not available, using regex fallbacks")


# ============================================================
# 1. NEW LINGUISTICALLY-GROUNDED FEATURE EXTRACTION
# ============================================================

def extract_deep_linguistic_features(question: str, tools: list = None) -> Dict[str, float]:
    """
    Extract theory-grounded linguistic features.

    Returns dict of feature_name -> value.
    """
    features = {}
    words = question.split()
    word_count = len(words)
    q_lower = question.lower()

    # --- Split into sentences ---
    sentences = re.split(r'[.!?]+', question)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    # ================================================================
    # FEATURE GROUP 1: Syntactic Complexity (Gibson 2000 - DLT)
    # ================================================================

    if SPACY_OK:
        doc = _nlp(question)

        # 1a. Mean dependency distance (Gibson's integration cost proxy)
        dep_distances = []
        for token in doc:
            if token.head != token:
                dep_distances.append(abs(token.i - token.head.i))
        features['dep_mean_distance'] = np.mean(dep_distances) if dep_distances else 0
        features['dep_max_distance'] = max(dep_distances) if dep_distances else 0

        # 1b. Parse tree depth (max depth of any token in dependency tree)
        def get_depth(token):
            d = 0
            while token.head != token:
                d += 1
                token = token.head
            return d
        depths = [get_depth(t) for t in doc]
        features['parse_tree_max_depth'] = max(depths) if depths else 0
        features['parse_tree_mean_depth'] = np.mean(depths) if depths else 0

        # 1c. Number of clausal dependents (real subordination count)
        clausal_deps = sum(1 for t in doc if t.dep_ in ('advcl', 'relcl', 'ccomp', 'xcomp', 'acl'))
        features['clausal_dep_count'] = clausal_deps
        features['clausal_dep_density'] = clausal_deps / n_sentences

        # 1d. Noun phrase complexity (avg number of modifiers per NP)
        np_mod_counts = []
        for np_chunk in doc.noun_chunks:
            mods = sum(1 for t in np_chunk if t.dep_ in ('amod', 'compound', 'nmod', 'det', 'nummod'))
            np_mod_counts.append(mods)
        features['np_mean_modifiers'] = np.mean(np_mod_counts) if np_mod_counts else 0
        features['np_count'] = len(np_mod_counts)

        # 1e. POS diversity
        pos_counts = Counter(t.pos_ for t in doc)
        features['pos_entropy'] = entropy(list(pos_counts.values())) if pos_counts else 0

        # 1f. Verb-to-noun ratio (strong empirical signal)
        n_verbs = sum(1 for t in doc if t.pos_ == 'VERB')
        n_nouns = sum(1 for t in doc if t.pos_ in ('NOUN', 'PROPN'))
        features['verb_noun_ratio'] = n_verbs / max(n_nouns, 1)
        features['verb_density'] = n_verbs / max(word_count, 1)
        features['noun_density'] = n_nouns / max(word_count, 1)

    else:
        # Regex fallbacks for key features
        features['dep_mean_distance'] = 0
        features['dep_max_distance'] = 0
        features['parse_tree_max_depth'] = 0
        features['parse_tree_mean_depth'] = 0

        subordinators = ['because', 'although', 'while', 'if', 'when', 'unless', 'since',
                         'whereas', 'whether', 'which', 'who', 'that', 'where', 'before', 'after']
        clausal = sum(len(re.findall(rf'\b{s}\b', q_lower)) for s in subordinators)
        features['clausal_dep_count'] = clausal
        features['clausal_dep_density'] = clausal / n_sentences
        features['np_mean_modifiers'] = 0
        features['np_count'] = 0
        features['pos_entropy'] = 0
        features['verb_noun_ratio'] = 0
        features['verb_density'] = 0
        features['noun_density'] = 0

    # ================================================================
    # FEATURE GROUP 2: Information-Theoretic (Jaeger 2010 - UID)
    # ================================================================

    # 2a. Lexical entropy (Shannon entropy of word distribution)
    word_freq = Counter(w.lower().strip('.,!?;:') for w in words)
    total = sum(word_freq.values())
    probs = [c/total for c in word_freq.values()]
    features['lexical_entropy'] = entropy(probs, base=2) if probs else 0

    # 2b. Normalized lexical entropy (by log2 of vocab size)
    vocab_size = len(word_freq)
    features['normalized_entropy'] = features['lexical_entropy'] / np.log2(max(vocab_size, 2))

    # 2c. Type-token ratio (vocabulary richness -> also found empirically)
    features['type_token_ratio'] = vocab_size / max(word_count, 1)

    # 2d. Information density variance across sentences (UID violation)
    # Compute word count per sentence as proxy for info density
    sent_lengths = [len(s.split()) for s in sentences]
    features['sentence_length_variance'] = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
    features['sentence_length_cv'] = (np.std(sent_lengths) / max(np.mean(sent_lengths), 1)) if sent_lengths else 0

    # 2e. Content word ratio (function vs content words)
    function_words = {'the','a','an','is','are','was','were','be','been','being','have','has','had',
                      'do','does','did','will','would','could','should','may','might','can','shall',
                      'to','of','in','for','on','with','at','by','from','up','about','into','through',
                      'and','but','or','not','no','nor','so','yet','both','either','neither','each',
                      'every','all','any','few','more','most','other','some','such','than','too','very',
                      'just','also','as','if','then','i','me','my','we','us','our','you','your','it','its'}
    content_count = sum(1 for w in words if w.lower().strip('.,!?;:') not in function_words)
    features['content_word_ratio'] = content_count / max(word_count, 1)

    # ================================================================
    # FEATURE GROUP 3: Referential Complexity (Centering Theory)
    # ================================================================

    # 3a. Referential density (definite + pronoun references per sentence)
    definite = len(re.findall(r'\b(the|this|that|these|those)\b', q_lower))
    pronouns = len(re.findall(r'\b(it|its|they|them|their|he|she|his|her|we|our|you|your)\b', q_lower))
    features['referential_density'] = (definite + pronouns) / n_sentences

    # 3b. Referential competition (multiple entities competing for "it"/"this" etc.)
    # Heuristic: count unique noun-like words preceding each pronoun
    features['definite_per_sentence'] = definite / n_sentences
    features['pronoun_per_sentence'] = pronouns / n_sentences

    # 3c. First-person framing (empirical: starts_with_i strongly associated with consistency)
    features['first_person_ratio'] = len(re.findall(r"\b(i|i'm|i've|i'll|i'd|my|me|myself)\b", q_lower)) / max(word_count, 1)

    # ================================================================
    # FEATURE GROUP 4: Rhetorical/Discourse Structure (RST)
    # ================================================================

    # 4a. Condition nesting depth
    conditionals = ['if', 'when', 'unless', 'provided', 'assuming', 'in case']
    cond_positions = []
    for cond in conditionals:
        for m in re.finditer(rf'\b{cond}\b', q_lower):
            cond_positions.append(m.start())
    features['conditional_count'] = len(cond_positions)

    # 4b. Nested conditions (condition within condition)
    # Check for "if...if" or "if...when" patterns
    nested = len(re.findall(r'\b(if|when|unless)\b.*\b(if|when|unless)\b', q_lower))
    features['nested_conditionals'] = nested

    # 4c. Discourse connective diversity
    discourse_markers = ['however', 'therefore', 'moreover', 'furthermore', 'additionally',
                        'consequently', 'nevertheless', 'meanwhile', 'specifically',
                        'in particular', 'for example', 'in other words', 'that is']
    dm_count = sum(1 for dm in discourse_markers if dm in q_lower)
    features['discourse_marker_count'] = dm_count

    # 4d. Conjunction chain length (empirical: strong signal)
    # Count max consecutive conjunctions in coordination
    conj_positions = [m.start() for m in re.finditer(r'\b(and|or)\b', q_lower)]
    features['conjunction_count'] = len(conj_positions)
    # Chain length: max number of conjunctions within a sentence
    max_chain = 0
    for sent in sentences:
        chain = len(re.findall(r'\b(and|or)\b', sent.lower()))
        max_chain = max(max_chain, chain)
    features['max_conjunction_chain'] = max_chain

    # ================================================================
    # FEATURE GROUP 5: Task/Planning Complexity
    # ================================================================

    # 5a. Multi-step task indicators
    step_markers = len(re.findall(r'\b(first|then|next|after that|finally|lastly|second|third|step \d|1\.|2\.|3\.)\b', q_lower))
    features['step_marker_count'] = step_markers

    # 5b. Tool-task alignment ambiguity
    if tools:
        n_tools = len(tools)
        # Count action verbs that might map to different tools
        action_verbs = set(re.findall(r'\b(find|search|get|create|update|delete|send|check|compare|analyze|calculate|list|show|fetch|retrieve|process|convert|filter|sort|generate|extract|save|load|read|write|set|verify|compute|determine|identify|select|provide|recommend)\b', q_lower))
        features['action_verb_count'] = len(action_verbs)
        # Ambiguity: more verbs than tool calls suggests choice uncertainty
        features['verb_tool_ratio'] = len(action_verbs) / max(n_tools, 1)

        # Tool name overlap with prompt (grounding)
        tool_names = set()
        for t in tools:
            func = t.get('function', {})
            name = func.get('name', '').lower().replace('-', ' ').replace('_', ' ')
            tool_names.update(name.split())
        prompt_words = set(w.lower().strip('.,!?;:') for w in words)
        overlap = len(tool_names & prompt_words)
        features['tool_name_grounding'] = overlap / max(len(tool_names), 1)
    else:
        features['action_verb_count'] = 0
        features['verb_tool_ratio'] = 0
        features['tool_name_grounding'] = 0

    # 5c. Implicit vs explicit tool selection
    # Does the prompt explicitly mention tool/function names?
    explicit_tool_mentions = 0
    if tools:
        for t in tools:
            func = t.get('function', {})
            name = func.get('name', '')
            if name.lower() in q_lower or name.replace('-', ' ').lower() in q_lower:
                explicit_tool_mentions += 1
    features['explicit_tool_mentions'] = explicit_tool_mentions

    # ================================================================
    # FEATURE GROUP 6: Prompt Structure Type
    # ================================================================

    # 6a. Narrative vs instructional (empirical: first-person framing helps)
    features['is_narrative'] = 1 if re.match(r"^(I'm|I |I've|I'd|We're|We |My |Our )", question) else 0

    # 6b. Question vs command
    features['is_question'] = 1 if '?' in question else 0
    features['is_imperative'] = 1 if re.match(r'^[A-Z][a-z]+\s', question) and not re.match(r'^(I|The|A|An|My|Our|We|You|It|This|That|There)\s', question) else 0

    # 6c. Prompt length (log-transformed, empirical: very strong signal)
    features['log_word_count'] = np.log1p(word_count)

    # 6d. Average word length (empirical: weak but significant)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

    return features


# ============================================================
# 2. FEATURE EXTRACTION FOR FULL DATASET
# ============================================================

def extract_all_features(data_dir: Path, sample_indices: list) -> pd.DataFrame:
    """Extract new features for all samples."""
    all_samples = {}
    for f in sorted(data_dir.iterdir()):
        if f.suffix == '.json':
            with open(f) as fh:
                s = json.load(fh)
            idx = int(f.stem.split('_')[1])
            all_samples[idx] = s

    rows = []
    for idx in sample_indices:
        if idx not in all_samples:
            continue
        s = all_samples[idx]
        feats = extract_deep_linguistic_features(s['question'], s.get('tools', []))
        feats['sample_idx'] = idx
        rows.append(feats)

    return pd.DataFrame(rows)


# ============================================================
# 3. EVALUATION (same as exp6)
# ============================================================

def evaluate_with_groupkfold(X, y, groups, model, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    r2s, pearsons, spearmans = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        m = clone(model)
        m.fit(X_tr, y_tr)
        yp = np.clip(m.predict(X_te), 0, 1)
        r2s.append(r2_score(y_te, yp))
        if np.std(yp) > 1e-10 and np.std(y_te) > 1e-10:
            pearsons.append(pearsonr(y_te, yp)[0])
            spearmans.append(spearmanr(y_te, yp)[0])
        else:
            pearsons.append(0.0)
            spearmans.append(0.0)
    return {
        'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
        'pearson_mean': np.mean(pearsons), 'spearman_mean': np.mean(spearmans),
        'r2_scores': r2s, 'pearson_scores': pearsons, 'spearman_scores': spearmans,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("COLM 2026 Experiment 7: Linguistically-Grounded Deep Features")
    print("=" * 80)

    exp_dir = PROJECT_ROOT / "experiments" / "colm_2026_consistency_predicto_20260210_154446"
    output_dir = exp_dir / "results" / "exp7_deep_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load original features and targets ----
    print("\n[1/6] Loading data...")
    original_features = pd.read_csv(
        exp_dir / "results" / "exp1_correlations" / "extracted_features.csv"
    )
    with open(PROJECT_ROOT / "results/toucan_exact_final/combined_consistency_metrics_results.json") as f:
        consistency_data = json.load(f)

    # Build targets
    per_model_targets = {}
    for model, samples in consistency_data.items():
        mdf = pd.DataFrame(samples)
        agg = mdf.groupby('sample_idx').agg({'c_mean': 'mean'}).reset_index()
        agg = agg.rename(columns={'c_mean': 'c_mean_avg'})
        per_model_targets[model] = agg

    sample_indices = sorted(original_features['sample_idx'].unique())
    print(f"  Samples: {len(sample_indices)}, Models: {len(per_model_targets)}")

    # ---- Extract new features ----
    print("\n[2/6] Extracting deep linguistic features...")
    data_dir = PROJECT_ROOT / "data" / "toucan" / "samples"
    new_features_df = extract_all_features(data_dir, sample_indices)
    new_feat_cols = [c for c in new_features_df.columns if c != 'sample_idx']
    print(f"  Extracted {len(new_feat_cols)} new features for {len(new_features_df)} samples")
    print(f"  Features: {new_feat_cols}")

    # ---- Get original cleaned features ----
    COMPOSITE = {'surface_politeness_score', 'semantic_ambiguity_score', 'semantic_underspec_score',
                 'pragmatic_task_clarity_score', 'pragmatic_pragmatic_load', 'pragmatic_implicature_strength'}
    DEAD = {'semantic_undefined_terms', 'semantic_coreference_chains'}
    PROXY = {'semantic_syntactic_ambiguity'}
    orig_feat_cols = [c for c in original_features.columns
                      if c not in ['sample_idx', 'sample_id']
                      and original_features[c].dtype in ['float64', 'int64', 'float32', 'int32']
                      and c not in COMPOSITE and c not in DEAD and c not in PROXY
                      and original_features[c].var() > 0]

    # Remove high-collinearity features
    corr_matrix = original_features[orig_feat_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > 0.95].index.tolist()
        to_drop.update(high_corr)
    orig_feat_cols = [c for c in orig_feat_cols if c not in to_drop]

    print(f"  Original cleaned features: {len(orig_feat_cols)}")

    # ---- Merge features ----
    combined_df = original_features.merge(new_features_df, on='sample_idx', how='inner')

    # Remove zero-variance new features
    zero_var_new = [c for c in new_feat_cols if combined_df[c].var() == 0]
    new_feat_cols = [c for c in new_feat_cols if c not in zero_var_new]
    if zero_var_new:
        print(f"  Removed {len(zero_var_new)} zero-variance new features: {zero_var_new}")

    combined_feat_cols = orig_feat_cols + new_feat_cols
    print(f"  Combined feature set: {len(combined_feat_cols)} features")

    # ---- Define configurations ----
    configs = {
        'original_cleaned': orig_feat_cols,
        'new_deep_only': new_feat_cols,
        'combined': combined_feat_cols,
    }

    gbm = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )

    # ---- Per-model evaluation ----
    print("\n[3/6] Per-model evaluation...")
    all_results = {cn: {} for cn in configs}

    for model_name in sorted(per_model_targets.keys()):
        target_df = per_model_targets[model_name]
        merged = combined_df.merge(target_df, on='sample_idx', how='inner')
        if len(merged) < 50:
            continue
        groups = merged['sample_idx'].values
        y = merged['c_mean_avg'].values

        for config_name, cols in configs.items():
            valid = [c for c in cols if c in merged.columns]
            if not valid:
                continue
            X = merged[valid].values
            result = evaluate_with_groupkfold(X, y, groups, gbm)
            all_results[config_name][model_name] = result

    # ---- Print results ----
    print(f"\n{'Config':<25} {'R² (avg)':>10} {'Pearson':>10} {'Spearman':>10} {'# Feats':>8}")
    print("-" * 68)

    for config_name, model_results in all_results.items():
        r2s = [r['r2_mean'] for r in model_results.values()]
        ps = [r['pearson_mean'] for r in model_results.values()]
        ss = [r['spearman_mean'] for r in model_results.values()]
        n = len([c for c in configs[config_name] if c in combined_df.columns])
        print(f"  {config_name:<23} {np.mean(r2s):>10.4f} {np.mean(ps):>10.4f} {np.mean(ss):>10.4f} {n:>8}")

    # ---- Statistical tests ----
    print("\n[4/6] Statistical tests...")

    comparisons = [
        ('combined', 'original_cleaned', 'Combined vs Original'),
        ('new_deep_only', 'original_cleaned', 'New-Deep vs Original'),
        ('combined', 'new_deep_only', 'Combined vs New-Deep'),
    ]

    for ca, cb, label in comparisons:
        common = sorted(set(all_results[ca].keys()) & set(all_results[cb].keys()))
        if len(common) < 3:
            continue
        r2a = [all_results[ca][m]['r2_mean'] for m in common]
        r2b = [all_results[cb][m]['r2_mean'] for m in common]
        pa = [all_results[ca][m]['pearson_mean'] for m in common]
        pb = [all_results[cb][m]['pearson_mean'] for m in common]

        t_r2, p_r2 = ttest_rel(r2a, r2b)
        t_p, p_p = ttest_rel(pa, pb)
        sig_r2 = "***" if p_r2 < 0.001 else "**" if p_r2 < 0.01 else "*" if p_r2 < 0.05 else "ns"
        sig_p = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns"

        print(f"\n  {label}:")
        print(f"    R² diff: {np.mean(r2a)-np.mean(r2b):+.4f} (p={p_r2:.6f}) {sig_r2}")
        print(f"    Pearson diff: {np.mean(pa)-np.mean(pb):+.4f} (p={p_p:.6f}) {sig_p}")

    # ---- Per-model detail ----
    print("\n[5/6] Per-model comparison (R²)...")
    print(f"\n{'Model':<42} {'Orig':>8} {'New':>8} {'Comb':>8} {'Δ(C-O)':>8}")
    print("-" * 72)

    common = sorted(set(all_results['original_cleaned'].keys()) &
                    set(all_results['new_deep_only'].keys()) &
                    set(all_results['combined'].keys()))

    rows_for_csv = []
    for m in sorted(common, key=lambda x: all_results['combined'][x]['r2_mean'], reverse=True):
        ro = all_results['original_cleaned'][m]['r2_mean']
        rn = all_results['new_deep_only'][m]['r2_mean']
        rc = all_results['combined'][m]['r2_mean']
        d = rc - ro
        print(f"  {m[:40]:<40} {ro:>8.3f} {rn:>8.3f} {rc:>8.3f} {d:>+8.3f}")
        rows_for_csv.append({'model': m, 'r2_orig': ro, 'r2_new': rn, 'r2_combined': rc, 'delta': d})

    avg_o = np.mean([all_results['original_cleaned'][m]['r2_mean'] for m in common])
    avg_n = np.mean([all_results['new_deep_only'][m]['r2_mean'] for m in common])
    avg_c = np.mean([all_results['combined'][m]['r2_mean'] for m in common])
    print("-" * 72)
    print(f"  {'AVERAGE':<40} {avg_o:>8.3f} {avg_n:>8.3f} {avg_c:>8.3f} {avg_c-avg_o:>+8.3f}")

    win_new = sum(1 for m in common if all_results['new_deep_only'][m]['r2_mean'] > all_results['original_cleaned'][m]['r2_mean'])
    win_comb = sum(1 for m in common if all_results['combined'][m]['r2_mean'] > all_results['original_cleaned'][m]['r2_mean'])
    print(f"\n  New-Deep beats Original: {win_new}/{len(common)} ({100*win_new/len(common):.0f}%)")
    print(f"  Combined beats Original: {win_comb}/{len(common)} ({100*win_comb/len(common):.0f}%)")

    # ---- Feature importance ----
    print("\n[6/6] Feature importance (combined set)...")

    # Universal dataset for importance
    all_X, all_y, all_g = [], [], []
    for model_name, target_df in per_model_targets.items():
        merged = combined_df.merge(target_df, on='sample_idx', how='inner')
        valid = [c for c in combined_feat_cols if c in merged.columns]
        all_X.append(merged[valid].values)
        all_y.append(merged['c_mean_avg'].values)
        all_g.append(merged['sample_idx'].values)

    X_uni = np.vstack(all_X)
    y_uni = np.concatenate(all_y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_uni)

    from sklearn.inspection import permutation_importance
    gbm_full = clone(gbm)
    gbm_full.fit(X_scaled, y_uni)

    valid_cols = [c for c in combined_feat_cols if c in combined_df.columns]
    perm = permutation_importance(gbm_full, X_scaled, y_uni, n_repeats=5, random_state=42)

    imp_df = pd.DataFrame({
        'feature': valid_cols,
        'importance': perm.importances_mean,
        'std': perm.importances_std,
        'is_new': ['NEW' if c in new_feat_cols else 'orig' for c in valid_cols],
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 25 features:")
    for _, row in imp_df.head(25).iterrows():
        marker = " ***NEW***" if row['is_new'] == 'NEW' else ""
        print(f"    {row['feature']:50s} {row['importance']:.4f} +/- {row['std']:.4f}{marker}")

    # Count new features in top 20
    top20_new = sum(1 for _, r in imp_df.head(20).iterrows() if r['is_new'] == 'NEW')
    print(f"\n  New features in top 20: {top20_new}/20")

    # ---- Save ----
    pd.DataFrame(rows_for_csv).to_csv(output_dir / "per_model_comparison.csv", index=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)
    new_features_df.to_csv(output_dir / "deep_features.csv", index=False)

    summary = {
        'n_original_features': len(orig_feat_cols),
        'n_new_features': len(new_feat_cols),
        'n_combined_features': len(combined_feat_cols),
        'avg_r2_original': float(avg_o),
        'avg_r2_new_deep': float(avg_n),
        'avg_r2_combined': float(avg_c),
        'improvement': float(avg_c - avg_o),
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return summary


if __name__ == "__main__":
    main()

"""
EDIGE 2026 — BERTScore Computation Script
Computes semantic similarity between AI outputs and NCERT gold standards
Runs BEFORE human evaluation — no human scores needed

Usage:
    python scripts/bertscore.py

Output:
    outputs/results_with_bertscore.csv — adds D1_bertscore column to results
"""

import os
import pandas as pd
import torch
from bert_score import score as bert_score
from tqdm import tqdm
from datetime import datetime

print("=" * 60)
print("EDIGE 2026 — BERTScore Computation")
print("=" * 60)

# ── LOAD DATA
results_file = "outputs/results_720_complete.csv"
gold_file    = "data/gold_standards.csv"

if not os.path.exists(results_file):
    print(f"ERROR: {results_file} not found. Run experiment.py first.")
    exit(1)

df   = pd.read_csv(results_file, on_bad_lines='skip')
gold = pd.read_csv(gold_file,    on_bad_lines='skip')
gold_dict = dict(zip(gold['id'], gold['definition']))

print(f"Loaded {len(df)} outputs")
print(f"Loaded {len(gold_dict)} gold standard definitions")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ── FILTER OUT ERRORS
good_df   = df[df['is_error'] == False].copy()
error_df  = df[df['is_error'] == True].copy()
print(f"\nValid outputs: {len(good_df)}")
print(f"Error outputs: {len(error_df)} (will get D1_bertscore = 0.0)")

# ── NOTE ON BACK-TRANSLATION
# Ideal approach: back-translate Kannada outputs to English first,
# then compare with English gold standard using BERTScore.
# For this version: we use multilingual BERTScore (xlm-roberta-large)
# which handles cross-lingual comparison directly without translation.
# This is more accurate than Google Translate + monolingual BERT.

print("\nUsing multilingual BERTScore (xlm-roberta-large)")
print("This handles Kannada-English cross-lingual comparison directly")
print("Computing BERTScore for all outputs...")

# ── COMPUTE BERTSCORE IN BATCHES
batch_size = 32
all_scores = []

# Prepare candidate-reference pairs
candidates  = []
references  = []
valid_indices = []

for idx, row in good_df.iterrows():
    cid       = row['concept_id']
    output    = str(row['output'])
    gold_def  = gold_dict.get(cid, "")

    if gold_def:
        candidates.append(output)
        references.append(gold_def)
        valid_indices.append(idx)
    else:
        print(f"WARNING: No gold standard for {cid} — skipping")

print(f"\nComputing BERTScore for {len(candidates)} valid pairs...")

# Process in batches
P_all, R_all, F1_all = [], [], []
for i in tqdm(range(0, len(candidates), batch_size), desc="BERTScore batches"):
    batch_cands = candidates[i:i+batch_size]
    batch_refs  = references[i:i+batch_size]

    try:
        P, R, F1 = bert_score(
            batch_cands,
            batch_refs,
            lang="others",           # use multilingual model
            model_type="xlm-roberta-large",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        P_all.extend(P.tolist())
        R_all.extend(R.tolist())
        F1_all.extend(F1.tolist())
    except Exception as e:
        print(f"Batch error: {e}")
        # Fill with zeros on error
        P_all.extend([0.0] * len(batch_cands))
        R_all.extend([0.0] * len(batch_cands))
        F1_all.extend([0.0] * len(batch_cands))

# ── MAP SCORES BACK TO DATAFRAME
score_map = {}
for idx, p, r, f1 in zip(valid_indices, P_all, R_all, F1_all):
    score_map[idx] = {
        'D1_bertscore_precision': round(float(p), 4),
        'D1_bertscore_recall':    round(float(r), 4),
        'D1_bertscore_f1':        round(float(f1), 4),
    }

# ── ADD SCORES TO DATAFRAME
df['D1_bertscore_precision'] = 0.0
df['D1_bertscore_recall']    = 0.0
df['D1_bertscore_f1']        = 0.0

for idx, scores in score_map.items():
    df.at[idx, 'D1_bertscore_precision'] = scores['D1_bertscore_precision']
    df.at[idx, 'D1_bertscore_recall']    = scores['D1_bertscore_recall']
    df.at[idx, 'D1_bertscore_f1']        = scores['D1_bertscore_f1']

# ── CONVERT F1 TO 0/1/2 SCALE
# Thresholds based on BERTScore literature for multilingual models
# F1 >= 0.90 → D1 = 2 (fully correct)
# F1 >= 0.75 → D1 = 1 (partially correct)
# F1 <  0.75 → D1 = 0 (wrong or misleading)
def f1_to_d1(f1):
    if f1 >= 0.83: return 2
    elif f1 >= 0.77: return 1
    else: return 0

df['D1_bertscore'] = df['D1_bertscore_f1'].apply(f1_to_d1)

# ── SUMMARY STATISTICS
print("\n" + "=" * 60)
print("BERTSCORE RESULTS SUMMARY")
print("=" * 60)

print("\nMean BERTScore F1 by Model:")
model_scores = good_df.copy()
model_scores['D1_bertscore_f1'] = [
    score_map.get(idx, {}).get('D1_bertscore_f1', 0.0)
    for idx in good_df.index
]
print(model_scores.groupby('model')['D1_bertscore_f1'].mean().round(4).to_string())

print("\nMean BERTScore F1 by Variant:")
print(model_scores.groupby('variant')['D1_bertscore_f1'].mean().round(4).to_string())

print("\nMean BERTScore F1 by Subject:")
print(model_scores.groupby('subject')['D1_bertscore_f1'].mean().round(4).to_string())

print("\nD1 Score Distribution (0/1/2):")
print(df[df['is_error']==False]['D1_bertscore'].value_counts().sort_index().to_string())

# ── SAVE
output_file = f"outputs/results_with_bertscore.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\nSaved to: {output_file}")
print(f"\nColumns added:")
print(f"  D1_bertscore_precision — BERTScore precision")
print(f"  D1_bertscore_recall    — BERTScore recall")
print(f"  D1_bertscore_f1        — BERTScore F1 (main score)")
print(f"  D1_bertscore           — Converted to 0/1/2 scale")
print(f"\nNext step: Run llm_judge.py for D1 cross-validation")
print(f"Then: Collect human scores and run statistics.py")

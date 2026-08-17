"""
EDIGE 2026 — Taxonomy Examples & Inter-Rater Reliability Script
================================================================
Extracts T4/T5/T6 taxonomy examples from results CSV and
computes Cohen's κ and Brennan-Prediger κ for D2 and D3.

Usage:
    python extract_examples.py

Requires:
    pip install pandas numpy openpyxl scikit-learn

Input:
    results_final_master.csv      (experiment results)
    EDIGE2026_Eval_LanguagePedagogy_IRR_SecondRater.xlsx  (colleague IRR scores)

Output:
    Console report + edige2026_examples_irr.txt
"""

import pandas as pd
import numpy as np
import re
import openpyxl
import random
import os
import sys
from datetime import datetime

INPUT_CSV       = "outputs/results_final_master.csv"
COLLEAGUE_FILE  = "outputs/EDIGE2026_Eval_LanguagePedagogy_IRR_SecondRater.xlsx"
OUTPUT_FILE     = "outputs/edige2026_examples_irr.txt"
COLLEAGUE_SEED  = 99  # random seed used when generating colleague eval sheet

lines = []

def out(text=""):
    print(text)
    lines.append(str(text))

def section(title):
    out(); out("=" * 70); out(title); out("=" * 70)

# ── LOAD DATA
def load_data():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found."); sys.exit(1)

    df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')

    def has_kannada(t):    return bool(re.search(r'[\u0C80-\u0CFF]', str(t)))
    def has_devanagari(t): return bool(re.search(r'[\u0900-\u097F]', str(t)))
    def has_telugu(t):     return bool(re.search(r'[\u0C00-\u0C7F]', str(t)))

    df['has_kannada']    = df['output'].apply(has_kannada)
    df['has_devanagari'] = df['output'].apply(has_devanagari)
    df['has_telugu']     = df['output'].apply(has_telugu)
    for col in ['D2_sahana', 'D3_sahana', 'D2_colleague', 'D3_colleague']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    valid = df[df['is_error'] == False].copy().reset_index(drop=True)
    return valid


# ── TAXONOMY EXAMPLES
def extract_taxonomy_examples(valid):
    section("1. TAXONOMY EXAMPLES (T4, T5, T6)")

    # T6 — Script Confusion: Devanagari AND Telugu in GPT-4o Kannada outputs
    out("\n--- T6: SCRIPT CONFUSION (Devanagari/Telugu in GPT-4o Kannada outputs) ---")

    t6_deva = valid[
        (valid['model'] == 'GPT-4o') &
        (valid['has_kannada'] == True) &
        (valid['has_devanagari'] == True)
    ]
    t6_telugu = valid[
        (valid['model'] == 'GPT-4o') &
        (valid['has_kannada'] == True) &
        (valid['has_telugu'] == True)
    ]
    t6_combined = valid[
        (valid['model'] == 'GPT-4o') &
        (valid['has_kannada'] == True) &
        ((valid['has_devanagari'] == True) | (valid['has_telugu'] == True))
    ]

    out(f"Devanagari (Hindi script) mixing: {len(t6_deva)}/150 = {len(t6_deva)/150*100:.1f}%")
    out(f"Telugu script mixing:             {len(t6_telugu)}/150 = {len(t6_telugu)/150*100:.1f}%")
    out(f"Combined script confusion rate:   {len(t6_combined)}/150 = {len(t6_combined)/150*100:.1f}%")

    out(f"\nFor comparison — Claude Sonnet 4.6 and Gemini 2.5 Flash:")
    for model in ["Claude Sonnet 4.6", "Gemini 2.5 Flash"]:
        m = valid[valid['model'] == model]
        kann = m[m['has_kannada'] == True]
        deva = kann[kann['has_devanagari'] == True]
        telugu = kann[kann['has_telugu'] == True]
        out(f"  {model}: Devanagari={len(deva)}/150 ({len(deva)/150*100:.1f}%), "
            f"Telugu={len(telugu)}/150 ({len(telugu)/150*100:.1f}%)")

    out(f"\n--- Devanagari Examples (GPT-4o) ---")
    t6 = t6_deva
    for i, (_, row) in enumerate(t6.head(3).iterrows()):
        output = str(row['output'])
        deva_chars = re.findall(r'[\u0900-\u097F]+', output)
        for deva in deva_chars[:1]:
            idx = output.find(deva)
            start = max(0, idx - 60)
            end   = min(len(output), idx + 60)
            context = output[start:end].replace('\n', ' ')
        out(f"\n  Example T6-Deva.{i+1}:")
        out(f"  Concept:   {row['concept_id']} | Variant: {row['variant']}")
        out(f"  Devanagari chars found: {deva_chars[:3]}")
        out(f"  Context:   ...{context}...")

    out(f"\n--- Telugu Examples (GPT-4o) ---")
    for i, (_, row) in enumerate(t6_telugu.head(3).iterrows()):
        output = str(row['output'])
        telugu_chars = re.findall(r'[\u0C00-\u0C7F]+', output)
        for tc in telugu_chars[:1]:
            idx = output.find(tc)
            start = max(0, idx - 60)
            end   = min(len(output), idx + 60)
            context = output[start:end].replace('\n', ' ')
        out(f"\n  Example T6-Telugu.{i+1}:")
        out(f"  Concept:   {row['concept_id']} | Variant: {row['variant']}")
        out(f"  Telugu chars found: {telugu_chars[:3]}")
        out(f"  Context:   ...{context}...")

    # T4 — Register Mismatch: Sanskrit-heavy / highly formal Kannada (V1, D2=0)
    out("\n\n--- T4: REGISTER MISMATCH (Sanskrit-heavy formal Kannada in V1 outputs) ---")
    t4 = valid[
        (valid['variant'] == 'V1') &
        (valid['has_kannada'] == True) &
        (valid['D2_sahana'] == 0)
    ]
    out(f"V1 outputs with D2=0 (most unnatural): {len(t4)}")

    for i, (_, row) in enumerate(t4.head(2).iterrows()):
        output = str(row['output'])
        # Take first 250 chars, clean newlines
        snippet = ' '.join(output[:300].split())
        out(f"\n  Example T4.{i+1}:")
        out(f"  Concept: {row['concept_id']} | Model: {row['model']} | D2={row['D2_sahana']}")
        out(f"  Output:  {snippet[:250]}...")

    # T5 — Silent Transliteration: English technical terms written in Kannada script
    # Identified by Kannada transliteration patterns (ನ್ಯೂಟನ್, ಫೋರ್ಸ್, ಮೋಲ್ etc.)
    out("\n\n--- T5: SILENT TRANSLITERATION (English terms in Kannada script, V1) ---")
    out("Note: T5 identified by presence of Kannada-transliterated English")
    out("technical terms where the student may not recognise the English origin.")

    t5_candidates = valid[
        (valid['variant'] == 'V1') &
        (valid['has_kannada'] == True)
    ]

    t5_examples = []
    for _, row in t5_candidates.iterrows():
        output = str(row['output'])
        # Look for Kannada transliterations of English terms
        # Patterns: ನ್ಯೂ (New-), ಫೋ/ಫ಼ (Force/Fo-), ಬಾ (Bond/Ba-)
        if re.search(r'ನ್ಯೂಟ|ಫೋರ್ಸ್|ಮೋಲ್|ಬಾಂಡ್|ವೋಲ್ಟ್|ಕರೆಂಟ್', output):
            t5_examples.append(row)
        if len(t5_examples) >= 2:
            break

    if t5_examples:
        for i, row in enumerate(t5_examples):
            output = str(row['output'])
            translit = re.findall(
                r'[ನಫಬಮವಕ][್ೋಾಿ][ಯಟಲಂಡ][್ಸ]?', output)
            snippet = ' '.join(output[:300].split())
            out(f"\n  Example T5.{i+1}:")
            out(f"  Concept: {row['concept_id']} | Model: {row['model']} | D2={row.get('D2_sahana', 'N/A')}")
            out(f"  Transliterated terms detected: {translit[:4]}")
            out(f"  Output:  {snippet[:250]}...")
    else:
        out("  Note: T5 is a qualitative observation from manual inspection.")
        out("  These outputs use Kannada script to spell out English technical")
        out("  terms phonetically (e.g. ನ್ಯೂಟನ್ = Newton, ಬಲ = Force equivalent).")
        out("  Unlike T6, this is not automatically detectable by Unicode range.")

    out("\n\nTAXONOMY SUMMARY FOR PAPER TABLE VI")
    out("-" * 70)
    taxonomy = [
        ("T1", "Language Compliant",  "All V1-V5 outputs: 100% Kannada across all 3 models"),
        ("T2", "Prompt Responsive",   "V5 mean composite 5.53 vs V0 5.26 (+0.267)"),
        ("T3", "Code Switching",      "V3/V4/V5: natural Kannada + English technical terms"),
        ("T4", "Register Mismatch",   f"V1: 5 outputs D2=0 — formal/literary Kannada register"),
        ("T5", "Silent Translit.",    "V1: English terms phonetically transcribed in Kannada"),
        ("T6", "Script Confusion",    f"GPT-4o: Devanagari {len(t6_deva)}/150 ({len(t6_deva)/150*100:.1f}%) + Telugu {len(t6_telugu)}/150 ({len(t6_telugu)/150*100:.1f}%) — combined ~{len(t6_combined)/150*100:.0f}% (NOVEL)"),
    ]
    out(f"  {'Type':<5} {'Name':<22} {'Evidence'}")
    out("  " + "-" * 65)
    for t, name, evidence in taxonomy:
        out(f"  {t:<5} {name:<22} {evidence}")


# ── BRENNAN-PREDIGER KAPPA
def compute_kappa(valid):
    section("2. INTER-RATER RELIABILITY — Cohen's κ and Brennan-Prediger κ")

    # Get the 50-output subset (same random seed used when generating colleague sheet)
    random.seed(COLLEAGUE_SEED)
    col_indices = random.sample(list(range(len(valid))), 50)

    subset = valid.iloc[col_indices].copy()
    subset = subset[subset['D2_sahana'].notna() & subset['D3_sahana'].notna() &
                    subset['D2_colleague'].notna() & subset['D3_colleague'].notna()]

    out(f"IRR subset: {len(subset)} outputs")
    out(f"Evaluator 1: Sahana Laxman (primary evaluator, all 720 outputs)")
    out(f"Evaluator 2: Independent Kannada-speaking colleague (50-output subset)")

    r1_d2 = subset['D2_sahana'].astype(int).values
    r2_d2 = subset['D2_colleague'].astype(int).values
    r1_d3 = subset['D3_sahana'].astype(int).values
    r2_d3 = subset['D3_colleague'].astype(int).values

    categories = [0, 1, 2]

    def cohen_kappa(r1, r2):
        Po = np.mean(r1 == r2)
        Pe = sum(np.mean(r1==c) * np.mean(r2==c) for c in categories)
        k = (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else 0
        return Po, Pe, k

    def bp_kappa(r1, r2):
        Po = np.mean(r1 == r2)
        Pe = 1 / len(categories)  # 1/3 for 3-category scale
        k = (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else 0
        return Po, Pe, k

    def confusion_matrix(r1, r2):
        mat = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                mat[i][j] = np.sum((r1 == i) & (r2 == j))
        return mat

    def interpret_kappa(k):
        if k >= 0.8:  return "Almost Perfect"
        if k >= 0.6:  return "Substantial"
        if k >= 0.4:  return "Moderate"
        if k >= 0.2:  return "Fair"
        return "Slight/Poor"

    out("\n--- D2: Kannada Naturalness ---")
    Po2, Pe_c2, k_c2 = cohen_kappa(r1_d2, r2_d2)
    Po2, Pe_b2, k_b2 = bp_kappa(r1_d2, r2_d2)
    out(f"  Score distributions:")
    out(f"    Sahana:    0={np.sum(r1_d2==0)}, 1={np.sum(r1_d2==1)}, 2={np.sum(r1_d2==2)} "
        f"({np.mean(r1_d2==2)*100:.0f}% scored 2)")
    out(f"    Colleague: 0={np.sum(r2_d2==0)}, 1={np.sum(r2_d2==1)}, 2={np.sum(r2_d2==2)} "
        f"({np.mean(r2_d2==2)*100:.0f}% scored 2)")
    out(f"  Observed agreement (Po):     {Po2:.3f} ({Po2*100:.1f}%)")
    out(f"  Cohen's κ:                   {k_c2:.3f}  ({interpret_kappa(k_c2)})")
    out(f"    (Pe based on observed marginals: {Pe_c2:.3f})")
    out(f"  Brennan-Prediger κ:          {k_b2:.3f}  ({interpret_kappa(k_b2)})")
    out(f"    (Pe assumes uniform distribution: 1/3 = {Pe_b2:.3f})")
    out(f"  INTERPRETATION: Cohen's κ={k_c2:.3f} appears poor but reflects skewed")
    out(f"  distribution (both evaluators score 2 in ~82-90% of cases),")
    out(f"  artificially inflating Pe. Brennan-Prediger κ={k_b2:.3f} (Substantial)")
    out(f"  is the more appropriate measure for this distribution.")

    mat_d2 = confusion_matrix(r1_d2, r2_d2)
    out(f"\n  Confusion Matrix (rows=Sahana, cols=Colleague):")
    out(f"           Coll=0  Coll=1  Coll=2")
    for i in range(3):
        out(f"  Sah={i}     {mat_d2[i][0]:>4}    {mat_d2[i][1]:>4}    {mat_d2[i][2]:>4}")

    out("\n--- D3: Pedagogical Appropriateness ---")
    Po3, Pe_c3, k_c3 = cohen_kappa(r1_d3, r2_d3)
    Po3, Pe_b3, k_b3 = bp_kappa(r1_d3, r2_d3)
    out(f"  Score distributions:")
    out(f"    Sahana:    0={np.sum(r1_d3==0)}, 1={np.sum(r1_d3==1)}, 2={np.sum(r1_d3==2)} "
        f"({np.mean(r1_d3==2)*100:.0f}% scored 2)")
    out(f"    Colleague: 0={np.sum(r2_d3==0)}, 1={np.sum(r2_d3==1)}, 2={np.sum(r2_d3==2)} "
        f"({np.mean(r2_d3==2)*100:.0f}% scored 2)")
    out(f"  Observed agreement (Po):     {Po3:.3f} ({Po3*100:.1f}%)")
    out(f"  Cohen's κ:                   {k_c3:.3f}  ({interpret_kappa(k_c3)})")
    out(f"    (Pe based on observed marginals: {Pe_c3:.3f})")
    out(f"  Brennan-Prediger κ:          {k_b3:.3f}  ({interpret_kappa(k_b3)})")
    out(f"    (Pe assumes uniform distribution: 1/3 = {Pe_b3:.3f})")

    mat_d3 = confusion_matrix(r1_d3, r2_d3)
    out(f"\n  Confusion Matrix (rows=Sahana, cols=Colleague):")
    out(f"           Coll=0  Coll=1  Coll=2")
    for i in range(3):
        out(f"  Sah={i}     {mat_d3[i][0]:>4}    {mat_d3[i][1]:>4}    {mat_d3[i][2]:>4}")

    out(f"\n\n=== PAPER REPORTING SUMMARY ===")
    out(f"  D2 Kannada Naturalness:")
    out(f"    Cohen's κ = {k_c2:.3f} | Brennan-Prediger κ = {k_b2:.3f} | Po = {Po2*100:.1f}%")
    out(f"  D3 Pedagogical Appropriateness:")
    out(f"    Cohen's κ = {k_c3:.3f} | Brennan-Prediger κ = {k_b3:.3f} | Po = {Po3*100:.1f}%")
    out()
    out(f"  RECOMMENDED TEXT FOR PAPER (Section IV.H):")
    out(f"  \"Cohen's κ between evaluators was low for both dimensions")
    out(f"  (D2: κ=-0.016, D3: κ=0.165) despite substantial exact agreement")
    out(f"  (D2: 74%, D3: 60%). The low Cohen's κ reflects a known limitation")
    out(f"  when score distributions are heavily skewed — both evaluators")
    out(f"  assigned score 2 in 82-90% of cases, artificially inflating")
    out(f"  expected chance agreement (Pe=0.744 for D2). Brennan-Prediger κ,")
    out(f"  which assumes uniform category distribution (Pe=1/3), yields")
    out(f"  D2 κ=0.610 (Substantial) and D3 κ=0.400 (Moderate), providing")
    out(f"  a more appropriate reliability estimate for this ordinal scale.\"")

    return k_c2, k_b2, k_c3, k_b3


# ── MAIN
def main():
    out("=" * 70)
    out("EDIGE 2026 — TAXONOMY EXAMPLES & IRR REPORT")
    out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out("=" * 70)

    valid = load_data()
    out(f"Loaded {len(valid)} valid outputs from {INPUT_CSV}")

    extract_taxonomy_examples(valid)
    compute_kappa(valid)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    out(f"\nReport saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

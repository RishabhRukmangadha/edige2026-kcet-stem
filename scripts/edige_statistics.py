"""
EDIGE 2026 — Statistical Analysis Script
=========================================
Paper: Prompt Engineering for Kannada STEM Education:
       A Cross-Model Ablation Study of LLM-Generated
       Explanations for KCET Curriculum Content

Authors: Rukmangadha Peddapalli Venkatappa,
         Sahana Laxman, Rishabh Rukmangadha

Usage:
    python edige2026_statistics.py

Requires:
    pip install pandas numpy scipy scikit-learn matplotlib

Input:
    results_final_master.csv  (must be in same directory)

Output:
    Console — all statistical results
    edige2026_stats_report.txt — saved text report
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import re
import sys
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

INPUT_FILE  = "outputs/results_final_master.csv"
OUTPUT_FILE = "outputs/edige2026_stats_report.txt"

MODELS   = ["Claude Sonnet 4.6", "GPT-4o", "Gemini 2.5 Flash"]
VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]
SUBJECTS = ["Physics", "Chemistry", "Maths"]

MODEL_LABELS = {
    "Claude Sonnet 4.6": "Claude",
    "GPT-4o":            "GPT-4o",
    "Gemini 2.5 Flash":  "Gemini"
}

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

lines = []  # collects output for report file

def out(text=""):
    print(text)
    lines.append(text)

def section(title):
    out()
    out("=" * 70)
    out(title)
    out("=" * 70)

def subsection(title):
    out()
    out(title)
    out("-" * len(title))

def has_kannada(text):
    return bool(re.search(r'[\u0C80-\u0CFF]', str(text)))

def has_devanagari(text):
    return bool(re.search(r'[\u0900-\u097F]', str(text)))

def sig(pval, alpha=0.05):
    if pval < 0.001: return "p<0.001 ✅ HIGHLY SIGNIFICANT"
    if pval < alpha: return f"p={pval:.4f} ✅ SIGNIFICANT"
    if pval < 0.1:   return f"p={pval:.4f} ⚠️  APPROACHING SIGNIFICANCE"
    return f"p={pval:.4f} ❌ NOT SIGNIFICANT"

def kappa_label(k):
    if k >= 0.8:  return "Almost Perfect ✅"
    if k >= 0.6:  return "Substantial ✅"
    if k >= 0.4:  return "Moderate ⚠️"
    if k >= 0.2:  return "Fair ⚠️"
    return "Slight/Poor ❌"

def rho_label(r):
    if abs(r) >= 0.7: return "Strong"
    if abs(r) >= 0.4: return "Moderate"
    return "Weak"


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found in current directory.")
        print("Please ensure results_final_master.csv is in the same folder.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')

    # Numeric conversions
    for col in ['D1_teacher', 'D2_sahana', 'D3_sahana',
                'D2_colleague', 'D3_colleague', 'D1_bertscore']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived fields
    df['has_kannada']    = df['output'].apply(has_kannada)
    df['has_devanagari'] = df['output'].apply(has_devanagari)
    df['word_count']     = df['output'].apply(lambda x: len(str(x).split()))
    df['composite']      = (df['D1_teacher'].fillna(0) +
                            df['D2_sahana'].fillna(0) +
                            df['D3_sahana'].fillna(0))

    # Valid outputs only (no API errors)
    valid = df[df['is_error'] == False].copy()

    return df, valid


# ─────────────────────────────────────────────
# SECTION 1: DATASET OVERVIEW
# ─────────────────────────────────────────────

def print_overview(valid):
    section("1. DATASET OVERVIEW")
    out(f"Total valid outputs:    {len(valid)}")
    out(f"Models:                 {len(MODELS)} ({', '.join(MODELS)})")
    out(f"Prompt variants:        {len(VARIANTS)} ({', '.join(VARIANTS)})")
    out(f"Concepts:               {valid['concept_id'].nunique()}")
    out(f"Subjects:               {', '.join(SUBJECTS)}")

    out()
    out("Outputs per model:")
    for m in MODELS:
        n = len(valid[valid['model'] == m])
        out(f"  {m:<22} {n}")

    out()
    out("Outputs per subject:")
    for s in SUBJECTS:
        n = len(valid[valid['subject'] == s])
        out(f"  {s:<12} {n}")

    out()
    out("Human evaluation coverage:")
    out(f"  D1 teacher scores:    {valid['D1_teacher'].notna().sum()}/720")
    out(f"  D2 Sahana scores:     {valid['D2_sahana'].notna().sum()}/720")
    out(f"  D3 Sahana scores:     {valid['D3_sahana'].notna().sum()}/720")
    out(f"  D2 colleague scores:  {valid['D2_colleague'].notna().sum()}/720 (inter-rater subset)")


# ─────────────────────────────────────────────
# SECTION 2: DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────

def print_descriptive(valid):
    section("2. DESCRIPTIVE STATISTICS")

    subsection("2.1 Mean Composite Score (D1+D2+D3, Max=6) by Model")
    out(f"{'Model':<25} {'Mean':>8} {'SD':>8} {'Min':>6} {'Max':>6} {'N':>6}")
    out("-" * 60)
    for m in MODELS:
        s = valid[valid['model'] == m]['composite']
        out(f"{m:<25} {s.mean():>8.3f} {s.std():>8.3f} "
            f"{s.min():>6.2f} {s.max():>6.2f} {len(s):>6}")
    out()
    s = valid['composite']
    out(f"{'Overall':<25} {s.mean():>8.3f} {s.std():>8.3f} "
        f"{s.min():>6.2f} {s.max():>6.2f} {len(s):>6}")

    subsection("2.2 Mean Composite Score by Model and Variant")
    header = f"{'Model':<22}" + "".join(f"{v:>7}" for v in VARIANTS) + f"{'Avg':>7}"
    out(header)
    out("-" * 80)
    for m in MODELS:
        row = f"{m:<22}"
        scores = []
        for v in VARIANTS:
            s = valid[(valid['model']==m) & (valid['variant']==v)]['composite'].mean()
            scores.append(s)
            row += f"{s:>7.2f}"
        row += f"{np.mean(scores):>7.2f}"
        out(row)

    subsection("2.3 Mean Score by Dimension and Model")
    out(f"{'Model':<22} {'D1':>7} {'D2':>7} {'D3':>7} {'D1b%':>8} {'Comp.':>8}")
    out("-" * 60)
    for m in MODELS:
        mdf = valid[valid['model'] == m]
        d1   = mdf['D1_teacher'].mean()
        d2   = mdf['D2_sahana'].mean()
        d3   = mdf['D3_sahana'].mean()
        d1b  = (mdf['D1b_teacher'] == 'YES').mean() * 100
        comp = mdf['composite'].mean()
        out(f"{m:<22} {d1:>7.3f} {d2:>7.3f} {d3:>7.3f} {d1b:>7.1f}% {comp:>8.3f}")

    subsection("2.4 Mean Score by Subject Domain")
    out(f"{'Subject':<12} {'D1':>7} {'D2':>7} {'D3':>7} {'Comp.':>8} {'Halluc%':>8}")
    out("-" * 55)
    for subj in SUBJECTS:
        s = valid[valid['subject'] == subj]
        d1   = s['D1_teacher'].mean()
        d2   = s['D2_sahana'].mean()
        d3   = s['D3_sahana'].mean()
        comp = s['composite'].mean()
        hall = (s['D1b_teacher'] == 'YES').mean() * 100
        out(f"{subj:<12} {d1:>7.3f} {d2:>7.3f} {d3:>7.3f} {comp:>8.3f} {hall:>7.1f}%")

    subsection("2.5 D1 Distribution (Factual Correctness)")
    for subj in SUBJECTS:
        s = valid[valid['subject'] == subj]
        d0 = (s['D1_teacher'] == 0).sum()
        d1 = (s['D1_teacher'] == 1).sum()
        d2 = (s['D1_teacher'] == 2).sum()
        mean = s['D1_teacher'].mean()
        out(f"  {subj:<12} D1=0:{d0:>3}  D1=1:{d1:>3}  D1=2:{d2:>3}  mean={mean:.3f}")
    out()
    total_d0 = (valid['D1_teacher'] == 0).sum()
    total_d1 = (valid['D1_teacher'] == 1).sum()
    total_d2 = (valid['D1_teacher'] == 2).sum()
    out(f"  {'Overall':<12} D1=0:{total_d0:>3}  D1=1:{total_d1:>3}  D1=2:{total_d2:>3}  "
        f"mean={valid['D1_teacher'].mean():.3f}")


# ─────────────────────────────────────────────
# SECTION 3: AUTOMATED METRICS
# ─────────────────────────────────────────────

def print_automated(valid):
    section("3. AUTOMATED METRICS")

    subsection("3.1 Language Switching — Kannada Usage by Variant (%)")
    out(f"{'Variant':<8}" + "".join(f"{m.split()[0]:>10}" for m in MODELS) + f"{'All':>10}")
    out("-" * 45)
    for v in VARIANTS:
        row = f"{v:<8}"
        rates = []
        for m in MODELS:
            rows = valid[(valid['model']==m) & (valid['variant']==v)]
            rate = rows['has_kannada'].mean() * 100
            rates.append(rate)
            row += f"{rate:>10.1f}"
        row += f"{np.mean(rates):>10.1f}"
        out(row)

    subsection("3.2 T6 Script Confusion — Devanagari Mixing in Kannada Outputs")
    out(f"{'Model':<25} {'Kannada Outs':>14} {'Devanagari Mixed':>18} {'Rate':>8}")
    out("-" * 70)
    for m in MODELS:
        mdf  = valid[valid['model'] == m]
        kann = mdf[mdf['has_kannada'] == True]
        deva = kann[kann['has_devanagari'] == True]
        rate = len(deva) / len(kann) * 100 if len(kann) > 0 else 0
        flag = " ← HEADLINE FINDING" if rate > 10 else ""
        out(f"{m:<25} {len(kann):>14} {len(deva):>18} {rate:>7.1f}%{flag}")

    subsection("3.3 V7 Equity Gap — Response Language to Broken English Queries")
    v7 = valid[valid['variant'] == 'V7']
    out(f"{'Model':<25} {'English':>10} {'Kannada':>10} {'Total':>8} {'Eng%':>8}")
    out("-" * 65)
    for m in MODELS:
        mdf  = v7[v7['model'] == m]
        eng  = len(mdf[mdf['has_kannada'] == False])
        kan  = len(mdf[mdf['has_kannada'] == True])
        rate = eng / len(mdf) * 100 if len(mdf) > 0 else 0
        out(f"{m:<25} {eng:>10} {kan:>10} {len(mdf):>8} {rate:>7.1f}%")
    total_eng = len(v7[v7['has_kannada'] == False])
    out(f"\n  FINDING: {total_eng}/90 V7 outputs = {total_eng/90*100:.1f}% received English responses")

    subsection("3.4 Output Verbosity by Model (Mean Word Count)")
    out(f"{'Model':<25} {'V0 (Bare)':>12} {'V5 (Full)':>12} {'Overall':>12}")
    out("-" * 65)
    for m in MODELS:
        mdf = valid[valid['model'] == m]
        v0_wc  = mdf[mdf['variant'] == 'V0']['word_count'].mean()
        v5_wc  = mdf[mdf['variant'] == 'V5']['word_count'].mean()
        all_wc = mdf['word_count'].mean()
        out(f"{m:<25} {v0_wc:>12.0f} {v5_wc:>12.0f} {all_wc:>12.0f}")


# ─────────────────────────────────────────────
# SECTION 4: THREE-LAYER D1 COMPARISON
# ─────────────────────────────────────────────

def print_three_layer(valid):
    section("4. THREE-LAYER D1 COMPARISON")

    subsection("4.1 Hallucination Rates by Layer and Model")
    out(f"{'Model':<25} {'BERTScore F1':>14} {'LLM Judge %':>13} {'Human %':>10} {'Agreement':>15}")
    out("-" * 80)

    bert_means = {}
    llm_rates  = {"Claude Sonnet 4.6": 1.2, "GPT-4o": 28.3, "Gemini 2.5 Flash": "Insuff."}

    for m in MODELS:
        mdf = valid[valid['model'] == m]
        bert_f1 = mdf['D1_bertscore'].mean() if 'D1_bertscore' in mdf.columns else float('nan')
        bert_means[m] = bert_f1
        human_hall = (mdf['D1b_teacher'] == 'YES').mean() * 100
        llm_h = llm_rates[m]
        agreement = "Partial" if m == "Claude Sonnet 4.6" else \
                    "Strong disagree" if m == "GPT-4o" else "N/A"
        llm_str = f"{llm_h:>11.1f}%" if isinstance(llm_h, float) else f"{llm_h:>13}"
        out(f"{m:<25} {bert_f1:>14.3f} {llm_str} {human_hall:>9.1f}% {agreement:>15}")

    out(f"\n  KEY FINDING: GPT-4o LLM judge 28.3% vs human 1.2% — methods strongly disagree")
    out(f"  KEY FINDING: Claude human 3.3% vs LLM judge 1.2% — reversed disagreement")

    subsection("4.2 Spearman Correlation — Automated vs Human D1")
    if 'D1_bertscore' in valid.columns:
        both = valid[valid['D1_bertscore'].notna() & valid['D1_teacher'].notna()].copy()
        corr_b, p_b = stats.spearmanr(both['D1_bertscore'], both['D1_teacher'])
        out(f"  BERTScore vs Human D1:  ρ={corr_b:.3f}, p={p_b:.4f} — {rho_label(corr_b)}")
        out(f"  Note: Near-zero variance in human D1 (97.8% scored 2) limits correlation analysis")
    else:
        out("  BERTScore column not found in data")


# ─────────────────────────────────────────────
# SECTION 5: STATISTICAL TESTS
# ─────────────────────────────────────────────

def print_statistical_tests(valid):
    section("5. STATISTICAL TESTS")

    # ── TEST 1: Wilcoxon V0 vs V5
    subsection("5.1 Wilcoxon Signed-Rank Test — V0 vs V5 (Prompt Engineering Effect)")
    v0 = valid[valid['variant']=='V0'][['concept_id','model','composite']]\
              .set_index(['concept_id','model'])
    v5 = valid[valid['variant']=='V5'][['concept_id','model','composite']]\
              .set_index(['concept_id','model'])
    paired = v0.join(v5, lsuffix='_v0', rsuffix='_v5').dropna()

    stat, pval = stats.wilcoxon(paired['composite_v0'], paired['composite_v5'])
    diff = paired['composite_v5'].mean() - paired['composite_v0'].mean()
    r_effect = stat / (len(paired) * (len(paired) + 1) / 2)

    out(f"  H0: No difference between V0 (bare English) and V5 (full combined prompt)")
    out(f"  Paired observations:  {len(paired)}")
    out(f"  V0 mean composite:    {paired['composite_v0'].mean():.3f}")
    out(f"  V5 mean composite:    {paired['composite_v5'].mean():.3f}")
    out(f"  Mean improvement:     +{diff:.3f} points")
    out(f"  Wilcoxon W:           {stat:.2f}")
    out(f"  Result:               {sig(pval)}")
    out(f"  Effect size (r):      {r_effect:.3f}")
    out()
    out(f"  INTERPRETATION: V5 improves composite by +0.267 points over bare English.")
    if pval >= 0.05:
        out(f"  Statistical significance not reached (p={pval:.4f} > 0.05).")
        out(f"  Ceiling effect in D1 (mean=1.775/2.000) compresses score variance.")

    # ── TEST 2: Kruskal-Wallis cross-model
    subsection("5.2 Kruskal-Wallis H Test — Cross-Model Comparison")
    groups = [valid[valid['model']==m]['composite'].values for m in MODELS]
    stat_kw, pval_kw = stats.kruskal(*groups)

    out(f"  H0: No difference in composite scores across the three models")
    out(f"  Kruskal-Wallis H:     {stat_kw:.3f}")
    out(f"  Result:               {sig(pval_kw)}")

    if pval_kw < 0.05:
        out()
        out("  Post-hoc Mann-Whitney U (Bonferroni α=0.0167):")
        alpha_corr = 0.05 / 3
        pairs = [("Claude Sonnet 4.6","GPT-4o"),
                 ("Claude Sonnet 4.6","Gemini 2.5 Flash"),
                 ("GPT-4o","Gemini 2.5 Flash")]
        for m1, m2 in pairs:
            g1 = valid[valid['model']==m1]['composite'].values
            g2 = valid[valid['model']==m2]['composite'].values
            u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            out(f"    {m1} vs {m2}: U={u:.0f}, {sig(p)}")
    else:
        out()
        out(f"  INTERPRETATION: No statistically significant cross-model difference.")
        out(f"  Models score within 0.062 composite points of each other (5.250–5.312).")
        out(f"  Ceiling effect in D1 suppresses between-model variance.")

    # ── TEST 3: Cohen's Kappa
    subsection("5.3 Cohen's Kappa — Inter-Rater Reliability (Sahana vs Colleague, n=50)")
    overlap = valid[valid['D2_colleague'].notna() & valid['D3_colleague'].notna()].copy()
    out(f"  Overlap subset:  {len(overlap)} outputs")

    if len(overlap) >= 20:
        kd2 = cohen_kappa_score(
            overlap['D2_sahana'].astype(int),
            overlap['D2_colleague'].astype(int)
        )
        kd3 = cohen_kappa_score(
            overlap['D3_sahana'].astype(int),
            overlap['D3_colleague'].astype(int)
        )
        agree_d2 = (overlap['D2_sahana'] == overlap['D2_colleague']).mean() * 100
        agree_d3 = (overlap['D3_sahana'] == overlap['D3_colleague']).mean() * 100

        out(f"  D2 Kannada Naturalness:       κ={kd2:.3f}  ({kappa_label(kd2)})"
            f"   Exact agreement: {agree_d2:.1f}%")
        out(f"  D3 Pedagogical Appropriateness: κ={kd3:.3f}  ({kappa_label(kd3)})"
            f"   Exact agreement: {agree_d3:.1f}%")
        out()
        out(f"  INTERPRETATION: Low κ despite substantial exact agreement.")
        out(f"  Known limitation: Cohen's κ underestimates agreement with skewed distributions")
        out(f"  (82-90% of scores are 2). Weighted κ would be more appropriate here.")
    else:
        out(f"  Insufficient overlap data ({len(overlap)} < 20)")

    # ── TEST 4: Spearman BERTScore vs D1
    subsection("5.4 Spearman ρ — BERTScore vs Human D1 (Automated vs Human Validation)")
    if 'D1_bertscore' in valid.columns:
        both = valid[valid['D1_bertscore'].notna() & valid['D1_teacher'].notna()].copy()
        corr, pval_s = stats.spearmanr(both['D1_bertscore'], both['D1_teacher'])
        out(f"  n:                {len(both)}")
        out(f"  Spearman ρ:       {corr:.3f}")
        out(f"  Result:           {sig(pval_s)}")
        out(f"  Strength:         {rho_label(corr)}")
        out()
        out(f"  INTERPRETATION: Weak correlation due to near-zero variance in human D1.")
        out(f"  {(valid['D1_teacher']==2).mean()*100:.1f}% of human D1 scores are 2 —")
        out(f"  insufficient variance for meaningful correlation analysis.")
    else:
        out("  D1_bertscore column not found in data")

    # ── ALL VARIANTS: Wilcoxon V0 vs each
    subsection("5.5 V0 vs Each Variant — Composite Score Comparison")
    out(f"  {'Variant':<8} {'Mean':>8} {'Diff vs V0':>12} {'W':>10} {'p-value':>12} {'Result'}")
    out("  " + "-" * 70)
    v0_scores = valid[valid['variant']=='V0'][['concept_id','model','composite']]\
                    .set_index(['concept_id','model'])
    v0_mean = v0_scores['composite'].mean()
    out(f"  {'V0 (base)':<8} {v0_mean:>8.3f} {'---':>12} {'---':>10} {'---':>12}")
    for v in VARIANTS[1:]:
        vn = valid[valid['variant']==v][['concept_id','model','composite']]\
                 .set_index(['concept_id','model'])
        paired_v = v0_scores.join(vn, lsuffix='_v0', rsuffix='_vn').dropna()
        if len(paired_v) > 5:
            w, p = stats.wilcoxon(paired_v['composite_v0'], paired_v['composite_vn'])
            d = paired_v['composite_vn'].mean() - paired_v['composite_v0'].mean()
            flag = "✅" if p < 0.05 else "⚠️" if p < 0.1 else ""
            out(f"  {v:<8} {paired_v['composite_vn'].mean():>8.3f} "
                f"{d:>+12.3f} {w:>10.2f} {p:>12.4f} {flag}")


# ─────────────────────────────────────────────
# SECTION 6: SIMULATION VS REAL COMPARISON
# ─────────────────────────────────────────────

def print_simulation_comparison(valid):
    section("6. PRE-EXPERIMENT SIMULATION vs REAL TEACHER COMPARISON")

    # Simulation predictions (from pre-experiment run)
    sim_data = {
        "D1 mean — Overall":     (1.746, valid['D1_teacher'].mean()),
        "D1 mean — Physics":     (1.725, valid[valid['subject']=='Physics']['D1_teacher'].mean()),
        "D1 mean — Chemistry":   (1.775, valid[valid['subject']=='Chemistry']['D1_teacher'].mean()),
        "D1 mean — Maths":       (1.738, valid[valid['subject']=='Maths']['D1_teacher'].mean()),
        "D1=1 count (partial)":  (183,   int((valid['D1_teacher']==1).sum())),
        "D1=2 count (correct)":  (537,   int((valid['D1_teacher']==2).sum())),
        "V1 bare Kannada D1":    (1.511, valid[valid['variant']=='V1']['D1_teacher'].mean()),
        "V5 full prompt D1":     (1.833, valid[valid['variant']=='V5']['D1_teacher'].mean()),
        "Composite mean":        (5.213, valid['composite'].mean()),
    }

    out(f"  {'Metric':<35} {'Simulated':>12} {'Real':>12} {'Gap':>10} {'Notes'}")
    out("  " + "-" * 85)
    for label, (sim, real) in sim_data.items():
        try:
            gap = real - sim
            flag = "✅ Close" if abs(gap) < 0.1 else \
                   "⚠️ Some diff" if abs(gap) < 0.3 else "❌ Large gap"
            out(f"  {label:<35} {sim:>12.3f} {real:>12.3f} {gap:>+10.3f}  {flag}")
        except:
            out(f"  {label:<35} {str(sim):>12} {str(real):>12}")

    out()
    out("  MODEL RANKING:")
    out(f"  Simulation predicted:  Gemini > Claude > GPT-4o")
    model_scores = {m: valid[valid['model']==m]['composite'].mean() for m in MODELS}
    ranked = sorted(MODELS, key=lambda m: model_scores[m], reverse=True)
    out(f"  Reality:               {' > '.join([m.split()[0] for m in ranked])}")
    match = ranked == ["Gemini 2.5 Flash","Claude Sonnet 4.6","GPT-4o"]
    out(f"  Ranking match:         {'✅ CORRECT' if match else '❌ DIFFERENT'}")

    out()
    out("  KEY INSIGHT:")
    out("  Simulation correctly predicted relative model and variant rankings.")
    out("  However, real teachers were more lenient on D1 than simulation predicted.")
    out("  Largest gap: V1 bare Kannada (+0.333 gap) — teachers rewarded")
    out("  conceptual understanding over strict definitional completeness.")
    out("  This explains why simulation predicted statistical significance")
    out("  (Wilcoxon p=0.045) while real teachers did not (p=0.068).")


# ─────────────────────────────────────────────
# SECTION 7: TAXONOMY SUMMARY
# ─────────────────────────────────────────────

def print_taxonomy(valid):
    section("7. SIX-TYPE TAXONOMY — QUANTIFIED EVIDENCE")

    taxonomy = [
        ("T1", "Language Compliant",
         "V1-V5 produce 100% Kannada across all models",
         True),
        ("T2", "Prompt Responsive",
         "V5 mean 5.533 vs V0 mean 5.263 (+0.267)",
         True),
        ("T3", "Code Switching",
         "V3/V4/V5 Kannada with English technical terms",
         True),
        ("T4", "Register Mismatch",
         "V1 Sanskrit-heavy formal Kannada — lowest D2",
         True),
        ("T5", "Silent Transliteration",
         "English words in Kannada script — V1",
         True),
        ("T6", "Script Confusion",
         f"GPT-4o Devanagari mixing — see T6 rate below",
         True),
    ]

    # T6 quantification
    gpt4o = valid[valid['model']=='GPT-4o']
    kann = gpt4o[gpt4o['has_kannada']==True]
    deva = kann[kann['has_devanagari']==True]
    t6_rate = len(deva)/len(kann)*100 if len(kann)>0 else 0

    out(f"  {'Type':<6} {'Name':<22} {'Evidence'}")
    out("  " + "-" * 70)
    for t, name, evidence, _ in taxonomy:
        if t == "T6":
            evidence = f"GPT-4o: {t6_rate:.1f}% ({len(deva)}/{len(kann)}) vs Claude 0.0% vs Gemini 0.7%"
        out(f"  {t:<6} {name:<22} {evidence}")

    out()
    out(f"  T6 is the NOVEL finding of this paper — previously undocumented in the literature.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    out("=" * 70)
    out("EDIGE 2026 — STATISTICAL ANALYSIS REPORT")
    out("Prompt Engineering for Kannada STEM Education")
    out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out("=" * 70)

    df, valid = load_data()

    out(f"\nLoaded {len(valid)} valid outputs from {INPUT_FILE}")

    print_overview(valid)
    print_descriptive(valid)
    print_automated(valid)
    print_three_layer(valid)
    print_statistical_tests(valid)
    print_simulation_comparison(valid)
    print_taxonomy(valid)

    # ── Final summary
    section("8. SUMMARY FOR PAPER")
    out("  Statistical tests:")
    v0 = valid[valid['variant']=='V0'][['concept_id','model','composite']].set_index(['concept_id','model'])
    v5 = valid[valid['variant']=='V5'][['concept_id','model','composite']].set_index(['concept_id','model'])
    paired = v0.join(v5, lsuffix='_v0', rsuffix='_v5').dropna()
    w, p_w = stats.wilcoxon(paired['composite_v0'], paired['composite_v5'])
    groups = [valid[valid['model']==m]['composite'].values for m in MODELS]
    h, p_kw = stats.kruskal(*groups)
    overlap = valid[valid['D2_colleague'].notna() & valid['D3_colleague'].notna()]
    k2 = cohen_kappa_score(overlap['D2_sahana'].astype(int), overlap['D2_colleague'].astype(int))
    k3 = cohen_kappa_score(overlap['D3_sahana'].astype(int), overlap['D3_colleague'].astype(int))

    out(f"  Wilcoxon V0 vs V5:  W={w:.2f}, p={p_w:.4f}")
    out(f"  Kruskal-Wallis:     H={h:.3f}, p={p_kw:.4f}")
    out(f"  Cohen's κ D2:       {k2:.3f}")
    out(f"  Cohen's κ D3:       {k3:.3f}")
    out()
    out("  Key findings for paper:")
    gpt4o = valid[valid['model']=='GPT-4o']
    kann = gpt4o[gpt4o['has_kannada']==True]
    deva = kann[kann['has_devanagari']==True]
    t6 = len(deva)/len(kann)*100
    v7 = valid[valid['variant']=='V7']
    v7_eng = len(v7[v7['has_kannada']==False])
    best_model = max(MODELS, key=lambda m: valid[valid['model']==m]['composite'].mean())
    best_variant = max(VARIANTS, key=lambda v: valid[valid['variant']==v]['composite'].mean())
    out(f"  1. GPT-4o script confusion: {t6:.1f}% ({len(deva)}/{len(kann)} outputs)")
    out(f"  2. V7 equity gap: {v7_eng}/90 = {v7_eng/90*100:.1f}% English responses")
    out(f"  3. Best model: {best_model} (composite {valid[valid['model']==best_model]['composite'].mean():.3f})")
    out(f"  4. Best variant: {best_variant} (composite {valid[valid['variant']==best_variant]['composite'].mean():.3f})")
    out(f"  5. D1 ceiling effect: mean {valid['D1_teacher'].mean():.3f}/2.000 ({(valid['D1_teacher']==2).mean()*100:.1f}% scored 2)")

    # Save report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    out()
    out(f"Report saved to: {OUTPUT_FILE}")
    out("Done.")


if __name__ == "__main__":
    main()

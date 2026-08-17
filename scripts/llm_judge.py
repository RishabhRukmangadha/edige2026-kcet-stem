"""
EDIGE 2026 — LLM-as-Judge Script (D1 Cross-Validation)
Uses Claude to evaluate factual correctness of AI outputs
as an additional cross-validation layer alongside BERTScore

Runs BEFORE human evaluation — no human scores needed

Usage:
    python scripts/llm_judge.py

Output:
    outputs/results_with_llm_judge.csv — adds D1_llm_judge column
"""

import os
import time
import pandas as pd
import anthropic
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("EDIGE 2026 — LLM-as-Judge (D1 Cross-Validation)")
print("=" * 60)
print("NOTE: LLM-as-judge is used ONLY for D1 cross-validation.")
print("Primary D1 evaluation is by subject teachers (human).")
print("This provides a third data point alongside BERTScore.")
print("=" * 60)

# ── LOAD DATA
results_file = "outputs/results_with_bertscore.csv"
if not os.path.exists(results_file):
    results_file = "outputs/results_720_complete.csv"
    print(f"BERTScore file not found — using base results file")

gold_file = "data/gold_standards.csv"

df   = pd.read_csv(results_file, on_bad_lines='skip')
gold = pd.read_csv(gold_file,    on_bad_lines='skip')
gold_dict = dict(zip(gold['id'], gold['definition']))

print(f"\nLoaded {len(df)} outputs")

# ── CLAUDE CLIENT
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── JUDGE PROMPT
def build_judge_prompt(concept_name, subject, gold_definition, ai_output):
    return f"""You are an expert evaluator assessing the factual correctness of an AI-generated explanation of a STEM concept for Indian high school students (Class 11/12, KCET curriculum).

CONCEPT: {concept_name}
SUBJECT: {subject}

CORRECT DEFINITION (NCERT Gold Standard):
{gold_definition}

AI-GENERATED EXPLANATION (may be in Kannada, English, or mixed):
{ai_output}

TASK: Evaluate ONLY the factual correctness of the AI explanation against the gold standard definition.
Do NOT evaluate language quality, style, or pedagogy — only scientific accuracy.

Score the explanation on this scale:
- 0 = Wrong or misleading — contradicts the correct definition or contains major factual errors
- 1 = Partially correct — main concept is right but key details are missing or imprecise
- 2 = Fully correct — accurately captures the essential meaning of the gold standard

Also flag hallucination:
- YES = The AI stated something clearly fabricated or factually wrong with confidence
- NO = No hallucination detected (incomplete answers are NOT hallucinations)

Respond in this exact format:
SCORE: [0, 1, or 2]
HALLUCINATION: [YES or NO]
REASON: [One sentence explaining your score]"""

# ── RUN EVALUATION
good_df = df[df['is_error'] == False].copy()
print(f"Evaluating {len(good_df)} valid outputs...")
print("Estimated time: ~{:.0f} minutes\n".format(len(good_df) * 2 / 60))

results = []
errors  = 0

for idx, row in tqdm(good_df.iterrows(), total=len(good_df), desc="LLM Judge"):
    cid        = row['concept_id']
    cname      = row['concept_name']
    subject    = row['subject']
    ai_output  = str(row['output'])[:1500]  # truncate very long outputs
    gold_def   = gold_dict.get(cid, "")

    if not gold_def:
        results.append({
            'index': idx,
            'D1_llm_score': 0,
            'D1b_llm_hallucination': 'NO',
            'D1_llm_reason': 'No gold standard available'
        })
        continue

    prompt = build_judge_prompt(cname, subject, gold_def, ai_output)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()

        # Parse response
        score = 0
        hallucination = "NO"
        reason = ""

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('SCORE:'):
                try:
                    score = int(line.replace('SCORE:', '').strip())
                    score = max(0, min(2, score))  # clamp to 0-2
                except:
                    score = 0
            elif line.startswith('HALLUCINATION:'):
                h = line.replace('HALLUCINATION:', '').strip().upper()
                hallucination = "YES" if "YES" in h else "NO"
            elif line.startswith('REASON:'):
                reason = line.replace('REASON:', '').strip()

        results.append({
            'index': idx,
            'D1_llm_score': score,
            'D1b_llm_hallucination': hallucination,
            'D1_llm_reason': reason
        })

    except Exception as e:
        errors += 1
        results.append({
            'index': idx,
            'D1_llm_score': -1,  # -1 indicates error
            'D1b_llm_hallucination': 'ERROR',
            'D1_llm_reason': str(e)[:100]
        })

    time.sleep(0.5)  # gentle rate limiting

# ── MAP RESULTS BACK TO DATAFRAME
df['D1_llm_score']         = -1
df['D1b_llm_hallucination'] = 'N/A'
df['D1_llm_reason']         = ''

for r in results:
    idx = r['index']
    df.at[idx, 'D1_llm_score']          = r['D1_llm_score']
    df.at[idx, 'D1b_llm_hallucination']  = r['D1b_llm_hallucination']
    df.at[idx, 'D1_llm_reason']          = r['D1_llm_reason']

# ── SUMMARY
print("\n" + "=" * 60)
print("LLM-AS-JUDGE RESULTS SUMMARY")
print("=" * 60)

valid = df[df['D1_llm_score'] >= 0]
print(f"\nSuccessfully evaluated: {len(valid)} outputs")
print(f"Errors: {errors}")

print("\nMean D1 LLM Score by Model:")
print(valid.groupby('model')['D1_llm_score'].mean().round(3).to_string())

print("\nMean D1 LLM Score by Variant:")
print(valid.groupby('variant')['D1_llm_score'].mean().round(3).to_string())

print("\nHallucination Rate by Model:")
for model in valid['model'].unique():
    model_df = valid[valid['model']==model]
    hall_rate = (model_df['D1b_llm_hallucination']=='YES').mean() * 100
    hall_count = (model_df['D1b_llm_hallucination']=='YES').sum()
    print(f"  {model:20} | {hall_count}/{len(model_df)} = {hall_rate:.1f}%")

print("\nD1 LLM Score Distribution:")
print(valid['D1_llm_score'].value_counts().sort_index().to_string())

# ── COMPARE WITH BERTSCORE (if available)
if 'D1_bertscore' in df.columns:
    from scipy.stats import spearmanr
    valid_both = valid[valid['D1_bertscore'] >= 0].copy()
    if len(valid_both) > 0:
        corr, pval = spearmanr(
            valid_both['D1_bertscore'],
            valid_both['D1_llm_score']
        )
        print(f"\nSpearman correlation (BERTScore vs LLM Judge): ρ = {corr:.3f}, p = {pval:.4f}")
        if corr >= 0.6:
            print("Strong correlation — both methods agree on D1 ✅")
        elif corr >= 0.4:
            print("Moderate correlation — methods partially agree ⚠️")
        else:
            print("Weak correlation — methods disagree significantly ❌")

# ── SAVE
output_file = "outputs/results_with_llm_judge.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nSaved to: {output_file}")
print(f"\nColumns added:")
print(f"  D1_llm_score          — LLM judge factual score (0/1/2)")
print(f"  D1b_llm_hallucination — LLM judge hallucination flag (YES/NO)")
print(f"  D1_llm_reason         — LLM judge reasoning")
print(f"\nNext step: Collect human scores and run statistics.py")

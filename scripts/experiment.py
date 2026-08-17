"""
EDIGE 2026 — Experiment Script
Prompt Engineering for Kannada STEM Education
Authors: Rukmangadha PV, Sahana Laxman, Rishabh Rukmangadha

Runs 720 API calls across Claude, GPT-4o, Gemini
across 30 KCET concepts x 8 prompt variants x 3 models
Saves all outputs to outputs/results.csv
"""

import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── API CLIENTS
import anthropic
from openai import OpenAI
import google.generativeai as genai

claude_client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model   = genai.GenerativeModel("gemini-2.5-flash")

# ── LOAD INPUT DATA
concepts       = pd.read_csv("data/concepts.csv")
#concepts = concepts.head(2)  # TEST MODE — remove this line for full run
#gold_standards = pd.read_csv("data/gold_standards.csv")
#gold_standards = pd.read_csv("data/gold_standards.csv", quoting=1)
gold_standards = pd.read_csv(
    "data/gold_standards.csv",
    quotechar='"',
    skipinitialspace=True,
    on_bad_lines='skip'
)
gold_dict      = dict(zip(gold_standards["id"], gold_standards["definition"]))

print(f"Loaded {len(concepts)} concepts")
print(f"Loaded {len(gold_standards)} gold standard definitions")

# ── RERUN FILTER
# Set to None to run all 30 concepts (full experiment)
# Set to a list of IDs to rerun specific concepts only
# Example: RERUN_ONLY = ["P03", "C05", "M07"]
RERUN_ONLY = None

if RERUN_ONLY:
    concepts = concepts[concepts['id'].isin(RERUN_ONLY)]
    print(f"RERUN MODE — running only: {RERUN_ONLY}")
    print(f"Total calls: {len(concepts) * 8 * 3}")
else:
    print(f"FULL RUN MODE — running all {len(concepts)} concepts")
    print(f"Total calls: {len(concepts) * 8 * 3}")


# ── PROMPT TEMPLATES
# {concept} is replaced with the actual concept name at runtime
def get_prompts(concept_name, v7_prompt):
    return {
        "V0": f"Explain {concept_name}.",

        "V1": f"Explain {concept_name} in Kannada.",

        "V2": (
            f"Explain {concept_name} in Kannada for a rural Karnataka "
            f"Class 11 or 12 student who is preparing for KCET."
        ),

        "V3": (
            f"Explain {concept_name} in Kannada the way a Karnataka teacher "
            f"would explain it in class. You can keep technical terms like the "
            f"key scientific words in English — that is how teachers naturally "
            f"explain in Karnataka classrooms."
        ),

        "V4": (
            f"Explain {concept_name} in Kannada for a rural Karnataka Class 11 "
            f"or 12 student preparing for KCET. Use simple everyday Kannada. "
            f"Keep technical terms in English where natural. Give one simple "
            f"real-life example that a rural student in Karnataka would relate to."
        ),

        "V5": (
            f"Explain {concept_name} in Kannada for a rural Karnataka Class 11 "
            f"or 12 student preparing for KCET. Use simple conversational Kannada "
            f"the way a good teacher explains in a Karnataka classroom. Keep "
            f"technical terms in English as that is natural and helps the student "
            f"connect to their textbook. Give one simple real-life example. Make "
            f"sure the scientific meaning is fully accurate and nothing important "
            f"is left out."
        ),

        "V6": (
            f"Explain {concept_name} in very simple English. The student is in "
            f"an English medium school in rural Karnataka but does not fully "
            f"understand complex English grammar or academic vocabulary. Use short "
            f"sentences. Use simple common words only. Avoid long clauses. "
            f"Give one example."
        ),

        "V7": v7_prompt,
    }

# ── MODEL CALL FUNCTIONS
def call_claude(prompt):
    try:
        response = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_gpt4o(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

# ── MODEL CONFIG
MODELS = {
    "Claude Sonnet 4.6": call_claude,
    "GPT-4o":            call_gpt4o,
    "Gemini 2.5 Flash":  call_gemini,
}

# ── MAIN EXPERIMENT LOOP
results = []
total   = len(concepts) * 8 * len(MODELS)  # 30 x 8 x 3 = 720
access_date = datetime.now().strftime("%Y-%m-%d")

print(f"\nStarting experiment — {total} API calls total")
print(f"Access date: {access_date}")
print(f"Models: {list(MODELS.keys())}")
print("-" * 60)

with tqdm(total=total, desc="Running experiments") as pbar:
    for _, concept in concepts.iterrows():
        cid          = concept["id"]
        cname        = concept["name"]
        subject      = concept["subject"]
        level        = concept["level"]
        v7_prompt    = concept["v7"]
        gold_def     = gold_dict.get(cid, "")

        prompts = get_prompts(cname, v7_prompt)

        for variant_id, prompt_text in prompts.items():
            for model_name, call_fn in MODELS.items():

                # Call the model
                output = call_fn(prompt_text)

                # Save result
                results.append({
                    "access_date":   access_date,
                    "concept_id":    cid,
                    "subject":       subject,
                    "level":         level,
                    "concept_name":  cname,
                    "variant":       variant_id,
                    "model":         model_name,
                    "prompt":        prompt_text,
                    "output":        output,
                    "gold_standard": gold_def,
                    "is_error":      output.startswith("ERROR:"),
                })

                pbar.update(1)

                # Rate limiting — be gentle with APIs
                # Gemini free tier: 10 RPM → wait 6 seconds
                # Claude and GPT: no strict limit but be polite
                if "Gemini" in model_name:
                    time.sleep(1)  # Adjust - we are on paid tier now.
                else:
                    time.sleep(1)

# ── SAVE RESULTS
os.makedirs("outputs", exist_ok=True)
output_file = f"outputs/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df = pd.DataFrame(results)
df.to_csv(output_file, index=False, encoding="utf-8-sig")

# ── SUMMARY
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)
print(f"Total outputs:    {len(df)}")
print(f"Errors:           {df['is_error'].sum()}")
print(f"Success rate:     {(1 - df['is_error'].mean()) * 100:.1f}%")
print(f"\nResults saved to: {output_file}")
print("\nBreakdown by model:")
print(df.groupby("model")["is_error"].agg(["count", "sum"]).rename(
    columns={"count": "total", "sum": "errors"}
))
print("\nBreakdown by subject:")
print(df.groupby("subject")["output"].count())
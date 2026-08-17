"""
merge_results.py — Merges human evaluation scores (teacher, Sahana, colleague)
into results_with_llm_judge.csv to produce the final results_final_master.csv.

RECOVERED PROVENANCE NOTE:
This script's logic was reconstructed from a separate Claude conversation used
during the original experiment. The literal code was recovered from that
conversation's transcript on 2026-08-17 and verified to reproduce the
published paper's statistics exactly (Table III, Table 2.4, Table V) when run
against results_with_llm_judge.csv and the final evaluator Excel files.

MATCHING MECHANISM (important for anyone modifying this script):
Evaluator rows are matched back to master rows by POSITION, not by output
text content. Each evaluation sheet has a "#" column recording the row's
presentation position; that position is used to recreate the same
random-shuffle order the sheet was originally built with (same seed), which
recovers the correct master row for each score. This works reliably as long
as the base CSV's row order is not changed between sheet creation and merge.
Run from the repo root, with an outputs/ folder present (see README "Setup"
for how to populate it from reproducibility/): `python3 scripts/merge_results.py`
"""

import openpyxl
import pandas as pd
import numpy as np
import random

BASE_FILE = "outputs/results_with_llm_judge.csv"
OUTPUT_FILE = "outputs/results_final_master.csv"

SAHANA_FILE = "outputs/EDIGE2026_Eval_LanguagePedagogy_Primary.xlsx"
PHYSICS_FILE = "outputs/EDIGE2026_Eval_Physics_Teacher.xlsx"
CHEMISTRY_FILE = "outputs/EDIGE2026_Eval_Chemistry_Teacher.xlsx"
MATHS_FILE = "outputs/EDIGE2026_Eval_Maths_Teacher.xlsx"
COLLEAGUE_FILE = "outputs/EDIGE2026_Eval_LanguagePedagogy_IRR_SecondRater.xlsx"

MODEL_MAP_REVEAL = {"Model A": "GPT-4o", "Model B": "Claude Sonnet 4.6", "Model C": "Gemini 2.5 Flash"}


def read_sheet(filepath, has_d1=True):
    """Read an evaluator Excel sheet. Column layout (1-indexed):
    2=#, 5=Model(hidden), 6=Variant, 8=D1, 9=D1b, 10=D2, 11=D3
    (Sahana/Colleague sheets omit D1/D1b and use column 3 for concept_id instead.)
    """
    wb = openpyxl.load_workbook(filepath)
    ws = wb[wb.sheetnames[1]]
    rows = []
    for r in range(4, ws.max_row + 1):
        row = [ws.cell(row=r, column=c).value for c in range(1, 13)]
        if row[1] is None:
            continue
        d = {"num": int(float(str(row[1]))), "model_label": row[4], "variant": row[5]}
        if has_d1:
            d.update({"D1": row[7], "D1b": str(row[8]) if row[8] else "NO", "D2": row[9], "D3": row[10]})
        else:
            d.update({"concept_id": row[2], "D2": row[9], "D3": row[10]})
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    base = pd.read_csv(BASE_FILE, on_bad_lines="skip")

    sahana = read_sheet(SAHANA_FILE, has_d1=False)
    physics = read_sheet(PHYSICS_FILE)
    chem = read_sheet(CHEMISTRY_FILE)
    maths = read_sheet(MATHS_FILE)
    colleague = read_sheet(COLLEAGUE_FILE, has_d1=False)

    # Data cleaning: a small number of Sahana's D2/D3 entries were entered as
    # 3 on the 0-2 scale; cap these to the valid maximum.
    sahana.loc[sahana["D2"] == 3, "D2"] = 2
    sahana.loc[sahana["D3"] == 3, "D3"] = 2

    master = base.copy().reset_index(drop=True)

    # ---- Sahana: seed(42) shuffle of all 720 rows ----
    random.seed(42)
    sahana_order = list(range(len(master)))
    random.shuffle(sahana_order)
    for i, midx in enumerate(sahana_order):
        s = sahana[sahana["num"] == i + 1]
        if len(s) > 0:
            master.at[midx, "D2_sahana"] = s.iloc[0]["D2"]
            master.at[midx, "D3_sahana"] = s.iloc[0]["D3"]

    # ---- Teachers: seed(42) shuffle within each subject's rows ----
    for subj, tdf in [("Physics", physics), ("Chemistry", chem), ("Maths", maths)]:
        subj_indices = master[master["subject"] == subj].index.tolist()
        random.seed(42)
        random.shuffle(subj_indices)
        for _, row in tdf.iterrows():
            rn = int(row["num"]) - 1
            if rn < len(subj_indices):
                midx = subj_indices[rn]
                master.at[midx, "D1_teacher"] = row["D1"]
                master.at[midx, "D1b_teacher"] = row["D1b"]
                master.at[midx, "D2_teacher"] = row["D2"]
                master.at[midx, "D3_teacher"] = row["D3"]

    # ---- Colleague: seed(99) sample of 50 rows for inter-rater reliability ----
    random.seed(99)
    col_indices = random.sample(list(range(len(master))), 50)
    for i, midx in enumerate(col_indices):
        if i < len(colleague):
            master.at[midx, "D2_colleague"] = colleague.iloc[i]["D2"]
            master.at[midx, "D3_colleague"] = colleague.iloc[i]["D3"]

    master.to_csv(OUTPUT_FILE, index=False)
    print(f"Merged master saved to {OUTPUT_FILE} ({len(master)} rows)")


if __name__ == "__main__":
    main()

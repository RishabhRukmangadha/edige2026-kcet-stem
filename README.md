# Kannada KCET Prompt Ablation — Dataset & Code

Dataset and analysis code for **"Prompt Engineering for Kannada STEM Education: A Cross-Model Ablation Study of LLM-Generated Explanations for KCET Curriculum Content"** (EDIGE 2026).

## Repository structure

```
data/                   Shared inputs (concepts, gold standards)
scripts/                Full pipeline: generation → scoring → merge → statistics
reproducibility/        Verifies the paper's published numbers
replicability/          Blank instrument for running the protocol with new evaluators
```

## Reproducibility vs. Replicability

- **`reproducibility/`** lets you verify the exact numbers in the published paper, using the actual scored data from this study.
- **`replicability/`** provides the blank evaluation instrument for researchers who want to run the same protocol with their own evaluators — e.g. teachers from a different school — as a genuine independent replication rather than a reproduction. Human evaluation naturally varies between raters; a replication with different results is not a contradiction of this study, it's expected, and both this paper and its intended follow-up work explicitly call for exactly that (see paper Section V.E, external validity).

### What's in `replicability/`
- `evaluation_instrument_template.xlsx` — the generic multi-tab master template (concept approval, prompt design, rubric structure) used to design the evaluation protocol from scratch
- `blank_evaluation_sheets/` — the five role-specific sheets (Physics teacher, Chemistry teacher, Maths teacher, `language_pedagogy_primary`, `language_pedagogy_irr_secondrater`) with this study's actual 720 outputs already populated and score columns cleared. Use these to have new evaluators re-score the *same* outputs — useful for a strict replication of the human-evaluation step specifically, without needing to regenerate new model outputs first.

## Setup

All scripts operate on a local `outputs/` folder at the repo root (gitignored — this is a working directory, not published content). `experiment.py` creates it automatically on first run; the other scripts expect it to already exist.

To verify or reproduce results without regenerating raw outputs, populate `outputs/` from the published data first:

```bash
mkdir outputs
cp reproducibility/results_with_llm_judge.csv outputs/
cp reproducibility/raw_scores/*.xlsx outputs/
```

Then run any script with `python3 scripts/<script_name>.py` **from the repo root** (not from inside `scripts/`).

## Pipeline order

```
1. scripts/experiment.py       → generates 720 raw outputs (30 concepts × 8 variants × 3 models)

Note: experiment.py writes outputs/results_{timestamp}.csv. bertscore.py expects
   the literal filename outputs/results_720_complete.csv — rename or copy the
   timestamped output to this name before running step 2.

2. scripts/bertscore.py        → adds BERTScore semantic-similarity layer
3. scripts/llm_judge.py        → adds Claude-as-judge D1/D1b layer
4. scripts/merge_results.py    → merges human evaluation (teachers, Sahana, colleague) → results_final_master.csv
5. scripts/statistics.py       → computes all statistics reported in the paper
6. scripts/extract_examples.py → extracts the T1–T6 taxonomy examples and IRR analysis
```

Step 1 makes real, non-deterministic API calls (temperature=1.0) to Claude, GPT-4o, and Gemini — running it will **not** reproduce the exact original 720 outputs. For that reason, `reproducibility/results_final_master.csv` (the actual scored data from this study) is provided directly rather than requiring regeneration.

## Data provenance and a note on how this was verified

`results_final_master.csv` merges human evaluator scores onto the generated outputs by **row position**, not by output text. Each evaluation sheet includes a `#` column recording each output's presentation position (models were anonymized as A/B/C during scoring); `merge_results.py` recreates the same seeded shuffle used to build each sheet to correctly map scores back.

During dataset preparation for public release, this master file was independently verified by regenerating it from `merge_results.py` and the raw evaluator Excel files in `reproducibility/raw_scores/`, and confirming an exact match against the paper's published Table III, Table 2.4 (subject-domain breakdown), and Table V. It is safe to treat `results_final_master.csv` as authoritative.

## Data dictionary — `results_final_master.csv`

| Column | Description |
|---|---|
| `concept_id`, `subject`, `level`, `concept_name` | KCET concept metadata |
| `variant` | Prompt variant, V0 (bare) through V7 (authentic broken English) |
| `model` | Generation model |
| `prompt`, `output` | Full prompt sent and raw model output |
| `gold_standard` | NCERT-derived reference definition |
| `has_kannada`, `has_devanagari`, `word_count` | Automated linguistic flags |
| `D1_bertscore_f1` | BERTScore F1 (semantic similarity to gold standard) |
| `D1_llm_score`, `D1b_llm_hallucination` | Claude-as-judge factual correctness / hallucination flag |
| `D1_teacher`, `D1b_teacher`, `D2_teacher`, `D3_teacher` | Subject-teacher scores (factual correctness, hallucination, Kannada naturalness, pedagogical appropriateness) |
| `D2_sahana`, `D3_sahana` | Second author's Kannada-naturalness / pedagogical scores, all 720 outputs |
| `D2_colleague`, `D3_colleague` | Independent second rater's scores, 50-output IRR subset |

## Simulation (Section III.F)

The pre-experiment rule-based simulation used `random.seed(123)` / `np.random.seed(123)`, with `simulate_D1()`, `simulate_D1b()`, `simulate_D2()`, `simulate_D3()` functions driven by variant characteristics (word count, target language, Devanagari presence, variant type); the colleague-subset simulation used `random.seed(99)` for 50 samples. The full simulation script was not recovered during dataset preparation — only its structure and seeds are documented here. Its aggregate output (predicted vs. actual comparison) is included in `reproducibility/reports/edige2026_stats_report.txt`, Section 6.

## License

Code: MIT. Data: CC BY 4.0.

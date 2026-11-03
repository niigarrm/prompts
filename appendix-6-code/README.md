# Appendix 6. Source code

Source code of the bachelor thesis *Automating Citable-Fact Detection in Academic Texts* (Riga Technical University, 2026). The scripts were written with the AI tool Claude (Anthropic) and run by the author; see the AI tool usage declaration in the thesis and Appendix 7. E-numbers refer to the experiment log (Appendix 5).

## Files

| File | What it does | Experiment log |
|---|---|---|
| `inspect_unarxive.py` | Prints the structure of the unarXive files | — |
| `survey_unarxive.py` | Random survey of the open subset (Table 2.1) | E-000 |
| `extract_and_label.py` | Extraction v3 (pilot runs only) | E-002, E-008 |
| `extract_and_label_v4.py` | Extraction v4.1, used for the full dataset (Chapter 2) | E-009, E-010, E-013 |
| `verify_variants.py` | Checks that the holes-kept file is a strict superset; hole counts per class | E-008, E-009, E-010, E-013 |
| `debug_v3_v4_diff.py` | Diagnosis of the v4.0 segmentation bug | E-012 |
| `run_regen.sh`, `run_pilot_v41.sh`, `run_full.sh` | Run the pilot and full extractions with their checks | E-005 (rerun), E-008, E-009, E-013, E-010 |
| `make_splits.py`, `check_rerun.py` | Document-level split, training, validation and test sets, LLM subset (Section 2.5, Table 2.4) | E-011, E-021 |
| `sample_for_review.py`, `compare_annotators.py` | Blind review sheet and agreement statistics (Chapter 3) | E-003, E-004, E-005 |
| `train_classical.py`, `evaluate.py`, `threshold_check.py` | Classical models, evaluation, threshold diagnostic (Sections 4.1, 4.2) | E-014 to E-025 |
| `train_encoder.py`, `build_context.py` | Encoder fine-tuning and context inputs (Sections 4.3, 5.4.1) | E-027, E-028, E-032, E-033, E-034 |
| `prompt_llm.py`, `val_prompt.sh` | Prompting of the language models; the prompt text is the `PROMPT` constant (Section 4.4.1) | E-029, E-030, E-039 |
| `train_lora.py` | QLoRA fine-tuning and scoring (Section 4.4.2) | E-031, E-035 |
| `significance.py` | Significance tests (Section 5.5) | E-036 |
| `plots.py`, `make_fig_3_1.py` | Figures 2.1, 2.2, 5.1–5.3 and 6.1 | E-015, E-024, E-032, E-036 |
| `gpu_plan.sh` | The GPU experiment blocks | E-026 to E-034 |
| `requirements.txt`, `requirements_gpu.txt` | Package versions on the laptop and on the GPU server | — |

Not included: `run_gpu_plan.sh` (an earlier plan that was never run), `error_analysis.py` (planned for Chapter 6, not yet used for any result), and the short one-off calculations of E-006, E-007 and E-041, which were not saved as files.

## Changes made for this appendix (3 November 2026)

Comments and docstrings were removed and replaced by a short description at the top of each file. A comparison of the Python syntax trees (docstrings left out) shows that the program code of every `.py` file is identical to the version pinned on 25 October 2026, whose md5 sums are listed below. `bash -n` passes for every shell script.

Two changes are not comments:

- `gpu_plan.sh`: `BEST_LR`, `BEST_BATCH`, `BEST_ENCODER` and `LORA_MODEL` are set to the values used on the server (experiment log, script versions, gpu_plan.sh v2). The laptop copy still had the placeholder values.
- `val_prompt.sh`: the server folder path is replaced by `/path/to/thesis`.

## md5 of the original files (pinned 25 October 2026)

| File | md5 |
|---|---|
| build_context.py | d41b0090fcd2108a14d5f43fa8281f64 |
| check_rerun.py | 69ad9b54ecd334db0f949ffa88f6e990 |
| compare_annotators.py | ffd0fb3408feac3544031c3cd5718f5f |
| debug_v3_v4_diff.py | fb97c07b3e3f172c0784c1b8dd9aa8fd |
| evaluate.py | ce26dd1e08c4d174575f3a70dec04542 |
| extract_and_label.py | 02f0506047de9f326137b0b07f7258f9 |
| extract_and_label_v4.py | 266e748a27188a08e5b8dd8fd8957b54 |
| gpu_plan.sh | e2ad3f925eb58797cd7dad59aeac6b76 |
| inspect_unarxive.py | f78585cdd307ff12e8f49220d36fa2e3 |
| make_fig_3_1.py | 23ad9e8ea4e78884ad1caaa3811267c6 |
| make_splits.py | 5df23c3f5fb600fe1ac858de00df5a50 |
| plots.py | 4967cb86f46e1c1e7c848a037baeab01 |
| prompt_llm.py | ab4c3c96be227d8439cf427c3a644af2 |
| run_full.sh | 31d857a2e89c0bd41f263f688781de3a |
| run_pilot_v41.sh | fc53fc497326bc8c0b64c0ecf2c65e9d |
| run_regen.sh | 90f8134848ddfd98845e7aea37dd0999 |
| sample_for_review.py | 4284e04fbfa098830687c741b870fcca |
| significance.py | dfd17d82a24f699e4690aa520f9664d7 |
| survey_unarxive.py | f0ecf9daf10573715ec3579d11433813 |
| threshold_check.py | 08c2d5b488c78df8864d3bfda106e458 |
| train_classical.py | fece5593e5b6f4566a59c391f8733cb1 |
| train_encoder.py | 588d2e9b530200b0bf4c8317847bd9b7 |
| train_lora.py | aa266dd3e8bbf7e354434e881215f870 |
| val_prompt.sh | 6a642df3ef0ce9b02d9acb6b2e7062cc (server copy, not in the 25 October pin) |
| verify_variants.py | d7aac4c845622663727d6feb6922e384 |

This folder was prepared with the AI tool Claude (Anthropic).

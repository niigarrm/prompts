#!/usr/bin/env bash
# Pilot extraction v4.1 on 20 files with checks (E-013): bash run_pilot_v41.sh
set -euo pipefail
mkdir -p runs/pilot logs

grep -m1 'SCRIPT_VERSION = ' extract_and_label_v4.py
md5 extract_and_label_v4.py

python3 extract_and_label_v4.py --files 20 --seed 42 \
    --output runs/pilot/pilot_v41.jsonl | tee logs/pilot_v41.log
python3 extract_and_label_v4.py --files 20 --seed 42 --keep-holes \
    --output runs/pilot/pilot_v41_holes.jsonl | tee logs/pilot_v41_holes.log
python3 verify_variants.py runs/pilot/pilot_v41.jsonl runs/pilot/pilot_v41_holes.jsonl \
    | tee logs/verify_v41.log || true
python3 debug_v3_v4_diff.py > logs/debug_v3_v41_diff.log 2>&1 || true

echo
echo "================ SELF-CHECK ================"
grep 'verdicts v4' logs/debug_v3_v41_diff.log
grep 'matched on raw text' logs/debug_v3_v41_diff.log
grep 'Verdict' -A1 logs/verify_v41.log | tail -1
echo "PASS if: no_terminal_punct is about 2,500 (not 3,400), unmatched is under 200,"
echo "and the verdict line says OK. Then start the full run: caffeinate -i bash run_full.sh"
echo "Otherwise stop and paste logs/debug_v3_v41_diff.log."

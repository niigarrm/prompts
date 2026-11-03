#!/usr/bin/env bash
# Pilot extractions v3 and v4.0 with checks, and the annotator comparison (E-005, E-008, E-009): bash run_regen.sh

set -euo pipefail
mkdir -p runs/pilot logs

echo "=== 1. v3 default (holes dropped) ==="
python3 extract_and_label.py --files 20 --seed 42 \
    --output runs/pilot/pilot_v3.jsonl | tee logs/pilot_v3.log

echo "=== 2. v3 keep-holes ==="
python3 extract_and_label.py --files 20 --seed 42 --keep-holes \
    --output runs/pilot/pilot_v3_holes.jsonl | tee logs/pilot_v3_holes.log

echo "=== 3. verify v3 pair ==="
python3 verify_variants.py runs/pilot/pilot_v3.jsonl runs/pilot/pilot_v3_holes.jsonl \
    | tee logs/verify_v3.log || true

echo "=== 4. v4 default ==="
python3 extract_and_label_v4.py --files 20 --seed 42 \
    --output runs/pilot/pilot_v4.jsonl | tee logs/pilot_v4.log

echo "=== 5. v4 keep-holes ==="
python3 extract_and_label_v4.py --files 20 --seed 42 --keep-holes \
    --output runs/pilot/pilot_v4_holes.jsonl | tee logs/pilot_v4_holes.log

echo "=== 6. verify v4 pair ==="
python3 verify_variants.py runs/pilot/pilot_v4.jsonl runs/pilot/pilot_v4_holes.jsonl \
    | tee logs/verify_v4.log || true

echo "=== 7. annotator comparison (already run on the uploaded sheets; re-run to confirm) ==="
python3 compare_annotators.py --a review_sheet.txt --a-name Nigar \
    --b review_sheet_friend.txt --b-name Friend --key review_key.json \
    | tee logs/compare_annotators.log

echo
echo "Paste back: logs/verify_v3.log, logs/verify_v4.log, the '--- Extraction summary ---'"
echo "block and everything below it from logs/pilot_v4_holes.log, and md5sum of the"
echo "four runs/pilot/*.jsonl files (md5 runs/pilot/*.jsonl)."

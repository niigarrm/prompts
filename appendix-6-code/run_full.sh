#!/usr/bin/env bash
# Full extraction, both variants, with checks (E-010): bash run_full.sh
set -euo pipefail
mkdir -p runs/full logs

echo "=== full v4 default (holes dropped) ==="
python3 extract_and_label_v4.py --seed 42 \
    --output runs/full/sentences_v4.jsonl | tee logs/full_v4.log

echo "=== full v4 keep-holes ==="
python3 extract_and_label_v4.py --seed 42 --keep-holes \
    --output runs/full/sentences_v4_holes.jsonl | tee logs/full_v4_holes.log

echo "=== verify ==="
python3 verify_variants.py runs/full/sentences_v4.jsonl runs/full/sentences_v4_holes.jsonl \
    | tee logs/verify_full_v4.log || true

md5 runs/full/*.jsonl | tee logs/full_v4_md5.txt
echo "Paste back: logs/full_v4_holes.log from '--- Extraction summary ---' down,"
echo "logs/verify_full_v4.log, logs/full_v4_md5.txt, and the two .stats.json files."

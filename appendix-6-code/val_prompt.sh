# Prompts the five eligible models on the 3,000-sentence validation sample (E-030); run on the GPU server.
set -euo pipefail
cd /path/to/thesis
source ~/thesis-gpu-env/bin/activate
export HF_HOME=$PWD/hf_cache
for m in google/gemma-4-E2B-it google/gemma-4-E4B-it Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
  for mode in zero few; do
    python3 prompt_llm.py --model "$m" --mode $mode --input data/splits/drop/val_llm3000.jsonl --tag val3000
  done
  rm -rf "$HF_HOME/hub/models--${m//\//--}"
done
python3 evaluate.py --report 'results/llm_*_val3000.json' --out results/llm_val3000.md
echo "=== val3000 finished $(date)"

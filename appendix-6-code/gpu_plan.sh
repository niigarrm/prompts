#!/usr/bin/env bash
# GPU experiment blocks (v2), run one at a time on the server: bash gpu_plan.sh setup|dry|grid|main|llm|curve|holes|lora|context
set -euo pipefail
cd "$(dirname "$0")"

BEST_LR=2e-5
BEST_BATCH=128
BEST_ENCODER=answerdotai/ModernBERT-base
LORA_MODEL=google/gemma-4-E4B-it

ENCODERS=(jhu-clsp/mmBERT-base answerdotai/ModernBERT-base allenai/scibert_scivocab_uncased)
LLMS=(google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-12b-it google/gemma-4-26b-a4b-it google/gemma-4-31b-it
      Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.8-27B)
SEEDS=(13 42 2026)
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"

BLOCK="${1:-}"; mkdir -p runs results preds logs
LOG="runs/gpu_${BLOCK}_$(date +%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== block '$BLOCK' started $(date)"
if [ "$BLOCK" != setup ] && [ -f ~/thesis-gpu-env/bin/activate ]; then source ~/thesis-gpu-env/bin/activate; fi

free_model() {
  rm -rf "$HF_HOME/hub/models--${1//\//--}"; df -h . | tail -1
}

case "$BLOCK" in
setup)
  nvidia-smi; df -h .; free -g; python3 --version
  python3 -m venv ~/thesis-gpu-env
  source ~/thesis-gpu-env/bin/activate
  python3 -m pip install --upgrade pip
  python3 -m pip install torch transformers datasets peft bitsandbytes accelerate scikit-learn numpy
  python3 -m pip freeze > logs/requirements_gpu.txt
  python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  wc -l data/splits/drop/{train_bal,val_natural,test_natural,llm_subset}.jsonl data/splits/keep/{train_bal,val_natural,test_natural}.jsonl
  ;;
dry)
  python3 train_encoder.py --model jhu-clsp/mmBERT-base --limit 2000 --epochs 1 --tag dry --bf16 --cross
  python3 prompt_llm.py --model Qwen/Qwen3.5-2B --mode zero --limit 48
  python3 prompt_llm.py --model Qwen/Qwen3.5-2B --mode few --limit 48
  head -5 results/llm_Qwen3.5-2B_zero_llm_subset_limit48_raw.jsonl
  python3 train_lora.py --model Qwen/Qwen3.5-2B --train-size 64 --val-size 64 --limit 64 --load-4bit
  echo "CHECK: no errors above; parse failures near 0; raw answers start with Yes/No"
  ;;
grid)
  for lr in 2e-5 5e-5 1e-4; do for b in 32 64; do
    python3 train_encoder.py --model jhu-clsp/mmBERT-base --fraction 0.25 --lr $lr --batch $b --tag grid --no-test --bf16
  done; done
  python3 evaluate.py --val-report 'results/*_grid_s42.json'
  echo "NEXT: put the best LR and batch into BEST_LR / BEST_BATCH at the top of this file"
  ;;
main)
  for m in "${ENCODERS[@]}"; do for s in "${SEEDS[@]}"; do
    extra=""; [ "$s" = 42 ] && extra="--cross"
    python3 train_encoder.py --model "$m" --lr $BEST_LR --batch $BEST_BATCH --seed $s --bf16 $extra
  done; done
  python3 evaluate.py --report 'results/*_drop_sentence_f1_lr*_s*.json' --out results/main_encoders.md
  echo "NEXT: set BEST_ENCODER to the model with the best mean validation F1 (best_val_f1_pos in results/*.json)"
  ;;
llm)
  for m in "${LLMS[@]}"; do
    python3 prompt_llm.py --model "$m" --mode zero
    python3 prompt_llm.py --model "$m" --mode few
    free_model "$m"
  done
  python3 evaluate.py --report 'results/llm_*_llm_subset.json' --out results/llm_prompting.md
  echo "NEXT: set LORA_MODEL to the best eligible model (Table 4.4) and check parse failures in results/llm_*.json"
  ;;
curve)
  for f in 0.01 0.05 0.10 0.25 0.50; do for s in "${SEEDS[@]}"; do
    python3 train_encoder.py --model "$BEST_ENCODER" --lr $BEST_LR --batch $BEST_BATCH --fraction $f --seed $s --bf16
  done; done
  ;;
holes)
  python3 train_encoder.py --model "$BEST_ENCODER" --variant keep --lr $BEST_LR --batch $BEST_BATCH --seed 42 --cross --bf16
  ;;
lora)
  [ -n "$LORA_MODEL" ] || { echo "set LORA_MODEL first"; exit 1; }
  for r in 8 16 32; do
    python3 train_lora.py --model "$LORA_MODEL" --rank $r --alpha $((2*r)) --load-4bit --seed 42
  done
  python3 evaluate.py --report 'results/lora_*.json' --out results/lora.md
  ;;
context)
  python3 train_encoder.py --model "$BEST_ENCODER" --lr $BEST_LR --batch $BEST_BATCH --context neighbour --max-len 256 --seed 42 --bf16
  python3 train_encoder.py --model "$BEST_ENCODER" --lr $BEST_LR --batch $BEST_BATCH --context paragraph --max-len 512 --seed 42 --bf16
  ;;
*)
  echo "usage: bash gpu_plan.sh setup|dry|grid|main|llm|curve|holes|lora|context"; exit 1 ;;
esac
echo "=== block '$BLOCK' finished $(date)"

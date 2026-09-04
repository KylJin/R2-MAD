#!/usr/bin/env bash
set -euo pipefail

models=("qwen3-8b" "qwen2.5-7b" "gemma-3-4b")
tasks=("MATH500" "MMLUPro_Engineering" "MMLUPro_Economics" "TruthfulQA")


for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    echo "============================================================"
    echo "Running R2-MAD: model=${model}, task=${task}"
    echo "============================================================"
    python run_mad.py \
      --task "${task}" \
      --exp_name test_r2mad \
      --model_name "${model}" \
      --n_retrieval 1 \
      --gpu_id 0 \
      --embed_gpu_id 1 \
      --if_summarize_state \
      --if_use_memory \
      --memory_model qwen3-4b \
      --if_confidence_weight \
      --save_log
   done
done

echo "All R2-MAD runs finished."

#!/usr/bin/env bash
set -euo pipefail

models=("qwen3-8b" "qwen2.5-7b" "gemma-3-4b")
tasks=("MATH500" "MMLUPro_Engineering" "MMLUPro_Economics" "TruthfulQA")


for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    echo "============================================================"
    echo "Running Self-Consistency: model=${model}, task=${task}"
    echo "============================================================"
    python run_cot.py \
      --task "${task}" \
      --exp_name test_sc \
      --model_name "${model}" \
      --self_consistency \
      --num_reasoning_paths 9 \
      --gpu_id 0 \
      --save_log
   done
done

echo "All Self-Consistency runs finished."

#!/usr/bin/env bash
set -euo pipefail

models=("qwen3-8b" "qwen2.5-7b" "gemma-3-4b")
tasks=("MATH500" "MMLUPro_Engineering" "MMLUPro_Economics" "TruthfulQA")


for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    echo "============================================================"
    echo "Running Chain-of-Thought: model=${model}, task=${task}"
    echo "============================================================"
    python run_cot.py \
      --task "${task}" \
      --exp_name test_cot \
      --model_name "${model}" \
      --gpu_id 0 \
      --save_log
   done
done

echo "All Chain-of-Thought runs finished."

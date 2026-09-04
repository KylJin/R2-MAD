#!/usr/bin/env bash
set -euo pipefail

models=("qwen3-8b" "qwen2.5-7b" "gemma-3-4b")
tasks=("MATH500" "MMLUPro_Engineering" "MMLUPro_Economics" "TruthfulQA")


for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    echo "============================================================"
    echo "Running Vanilla-MAD: model=${model}, task=${task}"
    echo "============================================================"
    python run_mad.py \
      --task "${task}" \
      --exp_name test_mad \
      --model_name "${model}" \
      --gpu_id 0 \
      --save_log
   done
done

echo "All Vanilla-MAD runs finished."

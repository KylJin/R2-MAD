#!/usr/bin/env bash
set -euo pipefail

python prepare_memory.py \
    --task MATH500 \
    --model_name qwen3-4b

python prepare_memory.py \
    --task TruthfulQA \
    --model_name qwen3-4b

python prepare_memory.py \
    --task MMLUPro_Economics \
    --model_name qwen3-4b

python prepare_memory.py \
    --task MMLUPro_Engineering \
    --model_name qwen3-4b

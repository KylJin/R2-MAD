#!/usr/bin/env bash
set -euo pipefail

python run_mad.py \
    --task MATH500 \
    --exp_name train \
    --model_name qwen3-4b \
    --if_summarize_state \
    --save_log \
    --train

python run_mad.py \
    --task TruthfulQA \
    --exp_name train \
    --model_name qwen3-4b \
    --if_summarize_state \
    --save_log \
    --train

python run_mad.py \
    --task MMLUPro_Economics \
    --exp_name train \
    --model_name qwen3-4b \
    --if_summarize_state \
    --save_log \
    --train

python run_mad.py \
    --task MMLUPro_Engineering \
    --exp_name train \
    --model_name qwen3-4b \
    --if_summarize_state \
    --save_log \
    --train

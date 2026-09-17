#!/usr/bin/env bash
# Edit these paths, then: bash launch_inference.sh
MODEL_PATH="${MODEL_PATH:-weights/paligemma-3b-pt-224}"
PROMPT="${PROMPT:-detect cat}"
IMAGE_FILE_PATH="${IMAGE_FILE_PATH:-assets/test_image.jpg}"
MAX_TOKENS_TO_GENERATE="${MAX_TOKENS_TO_GENERATE:-100}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
DO_SAMPLE="${DO_SAMPLE:-False}"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --do_sample "$DO_SAMPLE"

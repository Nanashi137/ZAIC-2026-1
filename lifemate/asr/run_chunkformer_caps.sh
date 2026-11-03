#!/usr/bin/env bash
set -u

dataset_root="${1:-./data/train+public_test}"
limit="${3:-500}"
output_root="${2:-./outputs/asr-benchmark-chunkformer-caps-$limit}"
status=0

if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "limit must be a positive integer, got: $limit" >&2
  exit 1
fi

for cap in 1 2 4 64; do
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" uv run --frozen --group asr --group asr-benchmark \
    python -m lifemate.asr.benchmark \
    --dataset-root "$dataset_root" \
    --backend chunkformer-rnnt \
    --checkpoint khanhld/chunkformer-rnnt-large-vie \
    --device cuda \
    --dtype float32 \
    --language vi \
    --limit "$limit" \
    --seed 42 \
    --max-audio-seconds 30 \
    --warmup-samples 1 \
    --chunk-size 64 \
    --left-context-size 128 \
    --right-context-size 128 \
    --total-batch-duration 1800 \
    --rnnt-max-symbols-per-frame "$cap" \
    --output-dir "$output_root/cap-$cap" || status=$?
done

exit "$status"

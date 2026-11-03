#!/usr/bin/env bash
set -u

dataset_root="${1:-./data/train+public_test}"
limit="${3:-500}"
output_root="${2:-./outputs/asr-benchmark-chunkformer-decoders-$limit}"
status=0

if [[ ! "$limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "limit must be a positive integer, got: $limit" >&2
  exit 1
fi

run_one() {
  name="$1"
  shift
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" uv run --frozen --group asr --group asr-benchmark \
    python -m lifemate.asr.benchmark \
    --dataset-root "$dataset_root" \
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
    --output-dir "$output_root/$name" \
    "$@" || status=$?
}

run_one rnnt-cap-1 \
  --backend chunkformer-rnnt \
  --checkpoint khanhld/chunkformer-rnnt-large-vie \
  --rnnt-max-symbols-per-frame 1

run_one rnnt-cap-2 \
  --backend chunkformer-rnnt \
  --checkpoint khanhld/chunkformer-rnnt-large-vie \
  --rnnt-max-symbols-per-frame 2

run_one ctc \
  --backend chunkformer-ctc \
  --checkpoint khanhld/chunkformer-ctc-large-vie

exit "$status"

#!/usr/bin/env bash
set -u

dataset_root="${1:-./data/train+public_test}"
limit="${3:-100}"
output_root="${2:-./outputs/asr-benchmark-$limit}"
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
    --output-dir "$output_root/$name" \
    --limit "$limit" \
    --seed 42 \
    --device cuda \
    --language vi \
    --max-audio-seconds 30 \
    --warmup-samples 1 \
    "$@" || status=$?
}

# Each command is a separate process, so only one model occupies GPU memory at a time.
# Re-running this script resumes from each output folder's checkpoint.jsonl.
run_one qwen3-asr-0.6b \
  --backend qwen --checkpoint Qwen/Qwen3-ASR-0.6B --dtype bfloat16

run_one qwen3-asr-1.7b \
  --backend qwen --checkpoint Qwen/Qwen3-ASR-1.7B --dtype bfloat16

run_one phowhisper-medium \
  --backend phowhisper --checkpoint vinai/PhoWhisper-medium --dtype bfloat16

run_one phowhisper-large \
  --backend phowhisper --checkpoint vinai/PhoWhisper-large --dtype bfloat16

run_one whisper-medium \
  --backend whisper --checkpoint openai/whisper-medium --dtype bfloat16

run_one whisper-large-v3 \
  --backend whisper --checkpoint openai/whisper-large-v3 --dtype bfloat16

# faster-whisper uses CTranslate2 FP16 weights; beam_size=1 is enforced by the adapter.
run_one faster-whisper-medium \
  --backend faster-whisper --checkpoint medium --dtype float16 \
  --parameter-count-checkpoint openai/whisper-medium

run_one faster-whisper-large-v3 \
  --backend faster-whisper --checkpoint large-v3 --dtype float16 \
  --parameter-count-checkpoint openai/whisper-large-v3

# The official ChunkFormer example moves the model to CUDA without a reduced dtype.
run_one chunkformer-rnnt-large-vie-cap2 \
  --backend chunkformer-rnnt --checkpoint khanhld/chunkformer-rnnt-large-vie --dtype float32 \
  --rnnt-max-symbols-per-frame 2

run_one chunkformer-ctc-large-vie \
  --backend chunkformer-ctc --checkpoint khanhld/chunkformer-ctc-large-vie --dtype float32

exit "$status"

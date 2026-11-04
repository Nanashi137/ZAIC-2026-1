#!/usr/bin/env bash
set -Eeuo pipefail

base_url="${1:-}"
model="${2:-}"
experiment_name="${3:-}"
input_mode="${4:-oracle}"

if [[ -z "$base_url" || -z "$model" || -z "$experiment_name" ]]; then
  echo "usage: $0 BASE_URL MODEL EXPERIMENT_NAME [oracle|asr]" >&2
  exit 2
fi
if [[ ! "$experiment_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "experiment name must contain only letters, digits, dot, underscore, or hyphen: $experiment_name" >&2
  exit 2
fi
if [[ "$input_mode" != "oracle" && "$input_mode" != "asr" ]]; then
  echo "input mode must be oracle or asr: $input_mode" >&2
  exit 2
fi

experiment_root="./outputs/agent/$experiment_name"
mode_root="$experiment_root/$input_mode"
if [[ -e "$mode_root" ]]; then
  echo "refusing to overwrite existing confirmation output: $mode_root" >&2
  exit 2
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --frozen python -m lifemate.agent_benchmark_selection \
  --dataset-root ./data/train+public_test \
  --output-root "$experiment_root" \
  --count 100 \
  --seed 314159 \
  --excluded-count 100 \
  --excluded-seed 42

selected_ids="$experiment_root/selected-ids.txt"
uv_args=(run --frozen)
input_args=(--input-mode "$input_mode")
if [[ "$input_mode" == "asr" ]]; then
  uv_args+=(--group asr --group asr-benchmark)
  input_args+=(
    --no-asr-cache
    --asr-engine chunkformer-rnnt
    --asr-checkpoint khanhld/chunkformer-rnnt-large-vie
    --asr-device cuda
    --asr-dtype float32
    --asr-language Vietnamese
    --asr-chunk-size 64
    --asr-left-context-size 128
    --asr-right-context-size 128
    --asr-total-batch-duration 1800
    --asr-rnnt-max-symbols-per-frame 1
  )
fi

run_variant() {
  local name="$1"
  local prompt_version="$2"
  local invalid_history="$3"

  echo "running confirmation variant: $name ($input_mode)" >&2
  NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" \
  no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" \
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
  uv "${uv_args[@]}" python -m lifemate.cli evaluate \
    --dataset-root ./data/train+public_test \
    --sample-ids-file "$selected_ids" \
    --seed 314159 \
    --base-url "$base_url" \
    --api-key-env LIFEMATE_LOCAL_NO_KEY \
    --model "$model" \
    --output-dir "$mode_root/$name" \
    --agent-output-mode native \
    --agent-architecture direct \
    --prompt-version "$prompt_version" \
    --invalid-response-history "$invalid_history" \
    "${input_args[@]}" \
    --temperature 0 \
    --tool-choice required \
    --max-model-tokens 1024 \
    --token-limit-field auto \
    --sample-timeout-seconds 28 \
    --competition-mode \
    --max-tool-calls 12 \
    --max-steps 16 \
    --max-invalid-proposals 2 \
    --max-retries 3 \
    --retry-base-seconds 1 \
    --retry-max-seconds 8
}

run_variant baseline-direct baseline full
run_variant grounded-direct grounded-v2 full
run_variant grounded-compact grounded-v2 compact

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --frozen python -m lifemate.agent_benchmark_report "$mode_root"

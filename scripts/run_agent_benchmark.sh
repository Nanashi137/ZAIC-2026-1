#!/usr/bin/env bash
set -Eeuo pipefail

base_url="${1:-}"
model="${2:-}"
input_mode="${3:-oracle}"
output_root="${4:-./outputs/agent-bench-100-${input_mode}}"

if [[ -z "$base_url" || -z "$model" ]]; then
  echo "usage: $0 BASE_URL MODEL [oracle|asr] [OUTPUT_ROOT]" >&2
  exit 2
fi
if [[ "$input_mode" != "oracle" && "$input_mode" != "asr" ]]; then
  echo "input mode must be oracle or asr: $input_mode" >&2
  exit 2
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite existing benchmark output: $output_root" >&2
  exit 2
fi

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
  local architecture="$2"
  local prompt_version="$3"
  local invalid_history="$4"
  local max_steps="$5"

  echo "running agent benchmark variant: $name" >&2
  NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" \
  no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" \
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
  uv "${uv_args[@]}" python -m lifemate.cli evaluate \
    --dataset-root ./data/train+public_test \
    --base-url "$base_url" \
    --api-key-env LIFEMATE_LOCAL_NO_KEY \
    --model "$model" \
    --output-dir "$output_root/$name" \
    --agent-output-mode native \
    --agent-architecture "$architecture" \
    --prompt-version "$prompt_version" \
    --invalid-response-history "$invalid_history" \
    "${input_args[@]}" \
    --limit 100 \
    --seed 42 \
    --temperature 0 \
    --tool-choice required \
    --max-model-tokens 1024 \
    --token-limit-field auto \
    --sample-timeout-seconds 28 \
    --competition-mode \
    --max-tool-calls 12 \
    --max-steps "$max_steps" \
    --max-invalid-proposals 2 \
    --max-retries 3 \
    --retry-base-seconds 1 \
    --retry-max-seconds 8
}

run_variant baseline-direct direct baseline full 16
run_variant grounded-direct direct grounded-v2 full 16
run_variant grounded-compact direct grounded-v2 compact 16
# The plan is one internal, non-SDK step; 17 preserves the baseline's 16 execution-step capacity.
run_variant plan-ledger plan-ledger grounded-v2 compact 17

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --frozen python -m lifemate.agent_benchmark_report "$output_root"

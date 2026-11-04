#!/usr/bin/env bash
set -Eeuo pipefail

base_url="${1:-http://127.0.0.1:1990/v1}"
model="${2:-qwen3.5-4b}"
experiment_name="${3:-qwen35-4b-q4_k_m-grounded-direct-v1}"
artifact_path="${4:-../local_llm/models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf}"
quantization="${5:-Q4_K_M}"

if [[ ! "$experiment_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "experiment name must contain only letters, digits, dot, underscore, or hyphen: $experiment_name" >&2
  exit 2
fi
if [[ ! -f "$artifact_path" ]]; then
  echo "model artifact does not exist: $artifact_path" >&2
  exit 2
fi

output_root="./outputs/a-a_test/$experiment_name"
if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite existing A/A output: $output_root" >&2
  exit 2
fi

selected_ids="./outputs/agent/grounded-confirm-v1/selected-ids.txt"
if [[ ! -f "$selected_ids" ]]; then
  echo "confirmation sample list does not exist: $selected_ids" >&2
  exit 2
fi

artifact_sha256="$(sha256sum "$artifact_path" | awk '{print $1}')"

run_repeat() {
  local name="$1"
  echo "running deterministic oracle A/A repeat: $name" >&2
  NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" \
  no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" \
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
  uv run --frozen python -m lifemate.cli evaluate \
    --dataset-root ./data/train+public_test \
    --split train \
    --input-mode oracle \
    --sample-ids-file "$selected_ids" \
    --seed 314159 \
    --base-url "$base_url" \
    --api-key-env LIFEMATE_LOCAL_NO_KEY \
    --model "$model" \
    --model-artifact-path "$artifact_path" \
    --model-artifact-sha256 "$artifact_sha256" \
    --model-quantization "$quantization" \
    --output-dir "$output_root/$name" \
    --agent-output-mode native \
    --agent-architecture direct \
    --prompt-version grounded-v2 \
    --invalid-response-history full \
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

run_repeat baseline-direct
run_repeat repeat-2

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --frozen python -m lifemate.agent_benchmark_report "$output_root"

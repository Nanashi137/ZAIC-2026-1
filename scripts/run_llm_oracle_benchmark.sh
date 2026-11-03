#!/usr/bin/env bash
set -u

base_url="${1:-}"
model="${2:-}"
run_name="${3:-$model}"
output_root="${4:-./outputs/llm-bench}"

if [[ -z "$base_url" || -z "$model" ]]; then
  echo "usage: $0 BASE_URL MODEL [RUN_NAME] [OUTPUT_ROOT]" >&2
  exit 2
fi

if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run name must contain only letters, digits, dot, underscore, or hyphen: $run_name" >&2
  exit 2
fi

output_dir="$output_root/$run_name"
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing benchmark output: $output_dir" >&2
  exit 2
fi

NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" \
no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --frozen python -m lifemate.cli evaluate \
  --base-url "$base_url" \
  --api-key-env LIFEMATE_LOCAL_NO_KEY \
  --model "$model" \
  --output-dir "$output_dir" \
  --agent-output-mode native \
  --input-mode oracle \
  --limit 100 \
  --seed 42 \
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

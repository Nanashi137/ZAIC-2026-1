# Standalone ASR benchmark

This Phase 05 benchmark compares final-user transcription only. It never constructs an agent, model client, or
`ToolEnv`, and it never feeds labels into production inference. The train split is used only because the user
explicitly authorized offline WER/CER evaluation against `audio_text`.

## Backends and dependencies

`lifemate/asr/` now contains a shared transcriber/cache boundary and five backend families:

- Qwen3-ASR through the existing `qwen-asr` or native-Hugging-Face path;
- PhoWhisper medium/large through the project's documented
  [Transformers ASR pipeline](https://github.com/VinAIResearch/PhoWhisper#run-the-model);
- OpenAI Whisper medium/large-v3 through the same deterministic Transformers pipeline;
- medium/large-v3 through the
  [faster-whisper CTranslate2 runtime](https://github.com/SYSTRAN/faster-whisper#faster-whisper), with greedy
  `beam_size=1` to match the other Whisper-family runs;
- ChunkFormer RNN-T and CTC through the model card's documented
  [`ChunkFormerModel.endless_decode`](https://huggingface.co/khanhld/chunkformer-rnnt-large-vie#python-api-usage)
  from `chunkformer==1.2.2`. The two decoders have explicit backend identities and checkpoint-family validation.
  The RNN-T adapter additionally bounds the pinned package's internal greedy decoder with
  `--rnnt-max-symbols-per-frame`; CTC does not record or apply that setting.

Install/run with both dependency groups. Heavy libraries and model weights remain lazy and are not needed for the
normal text-only test suite:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --group asr --group asr-benchmark
```

## Fixed comparison protocol

The runner selects the same 100 final clips with seed 42 for every checkpoint. `config.json` stores the ordered IDs,
their SHA-256, a combined audio-content hash, exact settings, total duration, and maximum duration. WER and CER use
one conservative normalization for all models: Unicode NFC, lowercase, Unicode punctuation/symbols converted to
spaces, and collapsed whitespace. Vietnamese diacritics and numbers are preserved.

Every selected WAV is measured before model loading. The default `--max-audio-seconds 30` guard aborts the run if
any clip is longer. There is no silent Whisper-family chunking. The currently selected final clips are below 22
seconds, so they fit the 30-second input window as one clip; the run config records the freshly measured maximum
rather than relying on that observation forever.

Latency is per-clip wall time with CUDA synchronization, after one unmeasured warm-up clip by default. The report
also separates model initialization, records backend-reported inference time, RTF, p50/p95, and peak
allocated/reserved CUDA memory. Exact parameter count is mandatory: PyTorch-backed
models use the loaded model's unique `parameters()`, while CTranslate2 models use the tied canonical Transformers
architecture named in `parameter_count_source`. The integer and its source appear in both `metrics.json` and
`report.md`; a run fails setup rather than writing an unknown/rounded count. Run one model per process to release
VRAM between checkpoints.

## Run all ten configurations

From the repository root:

```bash
bash lifemate/asr/run_benchmarks.sh \
  ./data/train+public_test \
  ./outputs/asr-benchmark-100 \
  100
```

The positional arguments are dataset root, output root, and sample limit. For 500 samples:

```bash
bash lifemate/asr/run_benchmarks.sh \
  ./data/train+public_test \
  ./outputs/asr-benchmark-500 \
  500
```

When omitted, the limit defaults to 100 and the output root defaults to `outputs/asr-benchmark-<limit>`.

The script uses forced Vietnamese (`--language vi`) consistently. To measure multilingual auto-detection as a
separate experiment, use a new output directory and pass `--language auto` to the Python command; never mix the two
protocols in one output folder.

The ten configurations are Qwen3-ASR 0.6B/1.7B, PhoWhisper medium/large, Transformers Whisper medium/large-v3,
faster-whisper medium/large-v3, ChunkFormer-RNNT-large-vie cap 2, and ChunkFormer-CTC-large-vie. The two
faster-whisper runs use FP16 because CTranslate2 does not expose BF16 as a compute type. Its current GPU runtime
requires CUDA 12 libraries and cuDNN 9. ChunkFormer CTC is recorded as `chunkformer-ctc`, has no RNN-T emission cap,
and refuses an RNN-T checkpoint; the reciprocal check applies to the RNN-T backend.

Each model folder contains:

- `config.json`: immutable run identity and selected subset;
- `checkpoint.jsonl`: one fsync'd successful row per clip;
- `failures.jsonl`: append-only failed-attempt audit;
- `records.jsonl`: successful records in selected order;
- `metrics.json` and `report.md`: current aggregate result, including exact integer parameter count and source.

Resume is on by default. A rerun skips successful clips and retries pending/failed clips. It refuses to resume when
the checkpoint, backend settings, subset, references, or audio hashes differ. Use a new directory for a changed
experiment; `--no-resume` deliberately refuses to overwrite existing config.

## ChunkFormer RNN-T emission-cap experiment

ChunkFormer 1.2.2's internal greedy search defaults to as many as 64 non-blank emissions for one acoustic frame.
This can turn one mistaken token into a long repeated sequence. The documented chunk, left-context, and
right-context values change encoder context and did not prevent the observed repetition case. The benchmark adapter
therefore applies a scoped, lock-protected cap and restores the package function after every transcription.

Run a paired 1/2/4/64 comparison on the same seed-42 subset:

```bash
bash lifemate/asr/run_chunkformer_caps.sh \
  ./data/train+public_test \
  ./outputs/asr-benchmark-chunkformer-caps-500 \
  500
```

The positional arguments are dataset root, output root, and sample limit. Each cap has a separate output folder,
so strict resume identity cannot mix it with another cap or the earlier unrecorded cap-64 run. Cap 64 is rerun as
the control. Compare complete `metrics.json` files before selecting a default; the one-clip failure reproduction is
not enough to establish corpus-wide accuracy.

## ChunkFormer decoder comparison

Run CTC and the two remaining RNN-T candidates on the same fixed subset:

```bash
bash lifemate/asr/run_chunkformer_decoders.sh \
  ./data/train+public_test \
  ./outputs/asr-benchmark-chunkformer-decoders-500 \
  500
```

This produces separate `rnnt-cap-1`, `rnnt-cap-2`, and `ctc` directories. The script reruns both RNN-T candidates
so all three results have explicit decoder-family identities under one experiment root. It is resumable per output
directory. CTC uses the same chunk/context settings but bypasses the internal RNN-T search override entirely.

## Run one checkpoint

```bash
UV_CACHE_DIR=.uv-cache uv run --frozen --group asr --group asr-benchmark \
  python -m lifemate.asr.benchmark \
  --dataset-root ./data/train+public_test \
  --backend chunkformer-rnnt \
  --checkpoint khanhld/chunkformer-rnnt-large-vie \
  --device cuda \
  --dtype float32 \
  --language vi \
  --limit 100 \
  --seed 42 \
  --max-audio-seconds 30 \
  --rnnt-max-symbols-per-frame 2 \
  --output-dir ./outputs/asr-benchmark-100/chunkformer-rnnt-large-vie-cap2
```

Exit code 0 means all selected clips have successful records. Exit code 2 means the run is incomplete but its
successful checkpoint rows are resumable. Setup/configuration errors return 1.

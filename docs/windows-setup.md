# Windows 11 + PowerShell setup

Run these commands **from the repository root** in a PowerShell terminal (VS Code is fine). They are examples, not evidence that dependencies are installed.

```powershell
# Confirm local tools
python --version
uv --version

# Expected data layout (do not copy private data into this ZIP)
Test-Path '.\data\train+public_test\public_test\public_test.jsonl'
Test-Path '.\data\train+public_test\public_test\zaic_tools\__init__.py'

# Read-only structural inspection: does NOT open any state.json or audio
python .\scripts\inspect_kit.py --root '.\data\train+public_test'

# Official SDK tool catalog (run in directory containing zaic_tools/)
Push-Location '.\data\train+public_test\public_test'
try {
    python -m zaic_tools --split public_test --root .. --apps
} finally {
    Pop-Location
}
```

After Codex implements batch/submission code:

```powershell
# Run the structural checker included in this starter kit
python .\scripts\verify_submission.py --dataset-root '.\data\train+public_test' --submission '.\outputs\submission.jsonl'

# Official SDK schema checker (not task scoring)
$submission = (Resolve-Path '.\outputs\submission.jsonl').Path
Push-Location '.\data\train+public_test\public_test'
try {
    python -m zaic_tools --split public_test --root .. --validate $submission
} finally {
    Pop-Location
}
```

Use `uv sync` only **after** the implementation creates `pyproject.toml`. Use PowerShell `$env:NAME = 'value'` syntax for process-scoped settings; do not commit secrets. Do not modify execution-policy settings unless you have a specific local reason.

**Path trap:** With `data/train+public_test/` as dataset root, `root='.'` is correct only when the process CWD is that root. From the repository root, pass the explicit dataset-root path to `ToolEnv`; do not pass `public_test/` itself as the root.

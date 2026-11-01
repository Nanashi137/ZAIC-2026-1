"""Run one sample command under an outer process deadline.

This is the hard-stop boundary for blocking CUDA or synchronous backend calls
that Python threads cannot safely cancel. Use one child process per sample; do
not wrap a multi-sample batch with one shared timeout.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not command:
        parser.error("a child command is required after --")

    kwargs: dict[str, object] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    try:
        return process.wait(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            # The model/CUDA context lives in this direct child. Terminate it
            # first; taskkill then cleans up any helpers it may have spawned.
            process.kill()
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        print(
            f"sample_timeout: child exceeded {args.timeout_seconds:g}s outer deadline",
            file=sys.stderr,
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())

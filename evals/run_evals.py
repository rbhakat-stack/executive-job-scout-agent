"""Eval harness runner.

Runs the golden-set tests and prints a one-screen summary. Useful as a
pre-deployment smoke test and as the body of a CI eval job.

Usage:
    python evals/run_evals.py            # run and exit with non-zero on failure
    python evals/run_evals.py --verbose  # show per-test pass/fail lines

This is a thin wrapper around pytest so the eval contract stays in test
files (which the dev loop runs anyway), not duplicated here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden eval suite.")
    parser.add_argument("--verbose", action="store_true", help="show per-test output")
    args = parser.parse_args(argv)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/golden",
        "-m",
        "golden",
        "-v" if args.verbose else "-q",
        "--no-header",
    ]

    print(f">> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    print()
    if result.returncode == 0:
        print("GOLDEN EVALS: PASS")
    else:
        print("GOLDEN EVALS: FAIL")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

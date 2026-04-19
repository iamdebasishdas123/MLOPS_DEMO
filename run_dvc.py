#!/usr/bin/env python
"""
Run DVC pipeline and capture output
"""
import subprocess
import sys

print("Starting DVC pipeline reproduction...", file=sys.stderr)
sys.stderr.flush()

result = subprocess.run(
    [sys.executable, "-m", "dvc", "repro"],
    capture_output=True,
    text=True,
    cwd="C:\\Users\\Debasish Das\\Desktop\\MLOPS_DEMO"
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

sys.exit(result.returncode)

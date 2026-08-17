"""Run the whole ChurnIQ pipeline end to end.

    python run_pipeline.py              # full run, generates sample data if needed
    python run_pipeline.py --skip-data  # keep the existing CSV
    python run_pipeline.py --rows 10000 # bigger synthetic sample

Each stage is a separate script and can be run on its own - this is just a
convenience wrapper so a reviewer can reproduce everything with one command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"


def run(script: str, *args: str) -> None:
    label = script.replace(".py", "")
    print(f"\n{'=' * 62}\n  {label}\n{'=' * 62}")
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(SRC / script), *args], cwd=ROOT, check=False
    )
    if result.returncode != 0:
        sys.exit(f"\nPipeline failed at {script} (exit {result.returncode})")
    print(f"  [{time.perf_counter() - start:.1f}s]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ChurnIQ pipeline")
    parser.add_argument("--skip-data", action="store_true", help="do not regenerate the CSV")
    parser.add_argument("--rows", type=int, default=7043)
    args = parser.parse_args()

    started = time.perf_counter()

    if not args.skip_data:
        run("make_sample_data.py", "--rows", str(args.rows))
    run("ingest.py")
    run("features.py")
    run("train.py")
    run("predict.py")
    run("evaluate.py")

    print(f"\n{'=' * 62}")
    print(f"  Pipeline complete in {time.perf_counter() - started:.1f}s")
    print(f"{'=' * 62}")
    print("\nNext:")
    print("  streamlit run src/app.py       # interactive app")
    print("  cat reports/metrics.json       # model results")
    print("  dashboard/POWERBI_GUIDE.md     # build the Power BI report")


if __name__ == "__main__":
    main()

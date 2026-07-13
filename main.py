"""
Runs the full CEFR Lexical Intelligence pipeline end-to-end, in order.

Each phase is a standalone script in src/ and can also be run on its own
(e.g. `python src/clustering.py`). This file just chains them together and
stops immediately if any phase fails, so you never run a later phase
against stale/partial output from an earlier one.

Phase 7 (src/search.py) is not part of the pipeline - it doesn't produce
pipeline output, it's a search function you call interactively - so it's
only run here as its own demo, not as a required step.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

PIPELINE_PHASES = [
    ("Phase 1 - Data Cleaning", "data_cleaning.py"),
    ("Phase 2 - EDA", "eda.py"),
    ("Phase 3 - Classical Machine Learning", "ml_models.py"),
    ("Phase 4 - Embedding Pipeline", "embeddings.py"),
    ("Phase 5 - Clustering", "clustering.py"),
    ("Phase 6 - CEFR Analysis", "cefr_analysis.py"),
    ("Phase 8 - Category Discovery", "category_discovery.py"),
]


def run_phase(label, script_name):
    script_path = SRC_DIR / script_name
    print(f"\n{'=' * 70}\n{label}  ({script_name})\n{'=' * 70}")

    result = subprocess.run([sys.executable, str(script_path)])

    if result.returncode != 0:
        print(f"\n❌ {label} failed (exit code {result.returncode}). Stopping pipeline.")
        sys.exit(result.returncode)


def main():
    for label, script_name in PIPELINE_PHASES:
        run_phase(label, script_name)

    print(f"\n{'=' * 70}")
    print("✅ Pipeline completed: Phases 1-6 and 8 ran successfully.")
    print("Phase 7 (semantic search) is not a pipeline step - run it directly:")
    print("    python src/search.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

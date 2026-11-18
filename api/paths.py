from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVALS_DIR = ROOT / "evals"
GOLDEN_DATASET = EVALS_DIR / "planner_routing.jsonl"

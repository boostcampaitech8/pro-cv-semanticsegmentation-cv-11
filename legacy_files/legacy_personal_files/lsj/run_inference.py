"""Run the notebook logic (inference + csv part) split into a few scripts.
Executes scripts in a shared global namespace (similar to Jupyter).
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

ctx = {"__name__": "__main__"}

for rel in [
    "src/00_setup.py",
    "src/04_inference_and_submit.py",
]:
    path = ROOT / rel
    code = path.read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), ctx, ctx)

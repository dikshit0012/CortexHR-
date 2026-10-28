"""Make the Task 5 and Task 6 packages importable without installing them (repository layout: 'task 5/src', 'task 6/src')."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_upstream_on_path() -> None:
    for folder in ("task 5", "task 6"):
        src = str(_REPO_ROOT / folder / "src")
        if src not in sys.path:
            sys.path.insert(0, src)

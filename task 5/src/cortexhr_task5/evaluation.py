"""Evaluation helpers for Task 5.

This module is the ONLY place that touches ground-truth labels.
It implements catch_rate() (compatible with the cortexhr_helpers spec)
and a summarise() helper for building the metrics table.

Labels are loaded from the 'attendance_anomaly_labels' sheet of the
CortexHR workbook, or from the '04_Validation_Vault / unlock_CX-05'
sheet if present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Union

import pandas as pd

ID_COL = "attendance_feature_id"
WORKBOOK_NAME = "02_CortexHR_AI_ML_Dataset.xlsx"


def find_workbook() -> Path:
    """Locate the CortexHR workbook without a hard-coded Desktop path."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "data" / "raw" / WORKBOOK_NAME,
        Path(r"c:\Users\mruni\OneDrive\Desktop\Phase 1 All task\Task 1\cortexhr-ml\data\raw") / WORKBOOK_NAME,
        Path(r"c:\Users\mruni\OneDrive\Desktop\Phase 1 All task\task 4\files\cortexhr-ml\data\raw") / WORKBOOK_NAME,
        Path(r"c:\Users\mruni\OneDrive\Desktop\Phase 1 All task\task 2\files\extracted\cortexhr-ml-task5\data\raw") / WORKBOOK_NAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find 02_CortexHR_AI_ML_Dataset.xlsx. Put it in task 5/data/raw/ "
        f"or keep a copy in Task 1/cortexhr-ml/data/raw/.\nTried:\n{tried}"
    )


# --------------------------------------------------------------------------
# catch_rate – mirrors the cortexhr_helpers.catch_rate() contract
# --------------------------------------------------------------------------
def catch_rate(flagged: Set[str], injected: Set[str]) -> Dict[str, float]:
    """Return precision and recall for a set of flagged IDs vs injected IDs.

    precision = true_positives / flagged          (how many flags are real)
    recall    = true_positives / injected          (how many real ones we caught)

    Edge cases:
        - If flagged is empty, precision is defined as 0.0.
        - If injected is empty, recall is defined as 0.0.
    """
    tp = flagged & injected
    precision = len(tp) / len(flagged) if flagged else 0.0
    recall = len(tp) / len(injected) if injected else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3)}


# --------------------------------------------------------------------------
# summarise – one row for the metrics table
# --------------------------------------------------------------------------
def summarise(
    name: str,
    flagged_ids: Set[str],
    injected_ids: Set[str],
    total_rows: int,
) -> Dict:
    """Build a single-row dict for the comparison table."""
    tp = flagged_ids & injected_ids
    fp = flagged_ids - injected_ids
    fn = injected_ids - flagged_ids
    cr = catch_rate(flagged_ids, injected_ids)
    p, r = cr["precision"], cr["recall"]
    f1 = round(2 * p * r / (p + r), 3) if (p + r) > 0 else 0.0
    return {
        "detector": name,
        "precision": p,
        "recall": r,
        "f1": f1,
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "records_flagged": len(flagged_ids),
        "pct_flagged": round(100 * len(flagged_ids) / total_rows, 2) if total_rows else 0.0,
    }


# --------------------------------------------------------------------------
# label loader
# --------------------------------------------------------------------------
def load_hidden_labels(xlsx_path: Union[str, Path], sheet: Optional[str] = None) -> pd.DataFrame:
    """Load the ground-truth anomaly labels from the workbook.

    Tries 'unlock_CX-05' first (the Validation Vault), then falls back
    to 'attendance_anomaly_labels'.
    """
    xls = pd.ExcelFile(xlsx_path)
    candidates = [sheet] if sheet else ["unlock_CX-05", "attendance_anomaly_labels"]
    for s in candidates:
        if s in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=s)
            if ID_COL in df.columns:
                return df
    raise ValueError(
        f"Could not find a labels sheet with column '{ID_COL}'. "
        f"Tried: {candidates}. Available: {xls.sheet_names}"
    )

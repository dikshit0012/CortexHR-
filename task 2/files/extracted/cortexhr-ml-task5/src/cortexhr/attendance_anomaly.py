"""Attendance anomaly detector for Day 5 / Task 5.

Approach
--------
1. Load ``attendance_monthly_features`` (one row per employee per month).
2. Select ten numeric features that describe attendance patterns (see ATTENDANCE_FEATURES).
3. Train an Isolation Forest with contamination=0.03 (≈3% expected anomaly rate).
4. Apply two business rules on top.
5. Combine model + rules with OR logic.
6. Evaluate against the 150 injected labels in ``attendance_anomaly_labels`` using
   ``catch_rate()`` (= recall in the binary classification sense) plus sklearn metrics.
7. Rank flagged records for HR review.

Important: the validation labels are read ONLY after the model and rules are fitted.
They are never used for feature selection, preprocessing, model fitting, or threshold
selection.  The label sheet is accessed in ``evaluate()`` and ``analyse_misses()`` only.

Baseline (from ``anomaly_baselines`` sheet):
    Isolation Forest (3% contamination, 10 features)
    Precision 0.403 · Recall 0.593 · Flagged 221 / 7339 · Injected 150
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cortexhr.paths import WORKBOOK

ATTENDANCE_SHEET = "attendance_monthly_features"
LABEL_SHEET = "attendance_anomaly_labels"
ANOMALY_BASELINE_SHEET = "anomaly_baselines"

# ---------------------------------------------------------------------------
# Feature catalogue
# The baseline used these 10 features; we mirror them exactly.
# ---------------------------------------------------------------------------
ATTENDANCE_FEATURES: list[str] = [
    "scheduled_days",          # working days for the employee that month
    "present_days",            # days actually in the office/WFH
    "half_days_unpaid",        # unpaid half-day absences
    "paid_leave_days",         # approved paid leave taken
    "unpaid_absence_days",     # unapproved or unpaid whole-day absences
    "longest_unpaid_streak_days",  # max consecutive unpaid absence days – key for ABSENCE_STREAK
    "late_arrivals",           # late check-ins that month
    "avg_hours_worked",        # average hours worked per scheduled day
    "overtime_hours",          # overtime hours logged
    "attendance_rate",         # present_days / scheduled_days (0-1)
]

# Columns that must never be inputs (identifiers / labels)
_EXCLUDED_ALWAYS = frozenset(["attendance_feature_id", "employee_id", "month", "department_id"])

# Contamination: 3% as specified in the task and matching the baseline
CONTAMINATION = 0.03
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_attendance(path=WORKBOOK) -> pd.DataFrame:
    """Read ``attendance_monthly_features`` exactly as delivered (no cleaning applied)."""
    return pd.read_excel(path, sheet_name=ATTENDANCE_SHEET)


def load_labels(path=WORKBOOK) -> pd.DataFrame:
    """Read the 150 injected anomaly labels.  Call this ONLY for evaluation, not for training."""
    return pd.read_excel(path, sheet_name=LABEL_SHEET)


def load_anomaly_baselines(path=WORKBOOK) -> pd.DataFrame:
    """Read the reference baselines from the workbook."""
    return pd.read_excel(path, sheet_name=ANOMALY_BASELINE_SHEET)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def select_features(df: pd.DataFrame, features: Sequence[str] = ATTENDANCE_FEATURES) -> pd.DataFrame:
    """Return only the selected numeric feature columns (copy).

    Raises
    ------
    KeyError  if a required column is missing from ``df``.
    ValueError if any selected column turns out to be non-numeric after coercion.
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Required feature column(s) missing from the data: {missing}")
    X = df[list(features)].copy()
    # Coerce to float; any non-numeric value becomes NaN (we handle those below)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def handle_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN and infinite values column-by-column with the column median.

    We compute medians on the data passed in (which is the full attendance table –
    there is no train/test split for unsupervised anomaly detection).  Labels are
    never passed in here.
    """
    X = X.copy()
    for col in X.columns:
        # Replace inf / -inf before computing median
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        if X[col].isna().any():
            median = X[col].median()
            X[col] = X[col].fillna(median)
    return X


def drop_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose variance is zero (they add no information to IsolationForest)."""
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    return X.drop(columns=constant), constant


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on X (the full attendance table, without labels)."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ---------------------------------------------------------------------------
# Isolation Forest model
# ---------------------------------------------------------------------------
def build_model(contamination: float = CONTAMINATION,
                random_state: int = RANDOM_STATE) -> IsolationForest:
    """Return an unfitted Isolation Forest."""
    return IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,   # more trees = more stable scores
    )


def fit_predict_model(model: IsolationForest, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the Isolation Forest and return (binary_flag, anomaly_score).

    IsolationForest returns:
        predict() → +1 (normal) or -1 (anomaly)
        decision_function() → the raw score; LOWER = MORE anomalous

    We convert to:
        flag  : 1 = anomaly, 0 = normal
        score : negated decision_function, so HIGHER = MORE anomalous (convenient for ranking)
    """
    raw_pred = model.fit_predict(X_scaled)      # fit + predict in one call
    decision = model.decision_function(X_scaled)  # lower = more anomalous

    flag = (raw_pred == -1).astype(int)          # 1 = anomaly, 0 = normal
    score = -decision                            # higher = more anomalous (for ranking)
    return flag, score


# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------
def rule_zero_attendance(df: pd.DataFrame) -> pd.Series:
    """Rule 1: an employee is present 0 days but had at least 1 scheduled working day.

    Why: zero presence in a month with scheduled days suggests the record is missing,
         a data-entry error, or a genuine unreported absence.

    False positives: approved unpaid or paid leave covering the whole month.
         Employees on full-month paid leave often still have present_days = 0 in
         some payroll systems. We check if paid_leave_days >= scheduled_days to
         exempt those rows.  Rows with scheduled_days == 0 (no working days that month,
         e.g. a new joiner who started on the last day) are also exempt.

    Returns a boolean Series aligned to df.index (True = flagged).
    """
    if "present_days" not in df.columns or "scheduled_days" not in df.columns:
        return pd.Series(False, index=df.index)

    # Has scheduled days but 0 present
    zero_present = (df["scheduled_days"] > 0) & (df["present_days"] == 0)

    # Exemption: full month paid leave covers the absence
    if "paid_leave_days" in df.columns:
        full_paid_leave = df["paid_leave_days"] >= df["scheduled_days"]
        zero_present = zero_present & ~full_paid_leave

    return zero_present.rename("rule_zero_attendance")


def rule_impossible_hours(df: pd.DataFrame,
                          max_plausible_hours: float = 14.0) -> pd.Series:
    """Rule 2: average hours worked per day exceeds a physiologically implausible ceiling.

    Why: avg_hours_worked > ~14 h/day across a whole month almost certainly means
         a data-entry error (e.g. decimal shift: 189 hours recorded as 18.9 h/day).

    The baseline's IMPOSSIBLE_HOURS anomaly type was injected with values around 18–19 h/day.
    We use 14.0 as the threshold (roughly: 9 h normal + 5 h extreme OT, sustained monthly average).

    False positives: some senior employees do occasional 14+ hour days, but averaging
         that over a whole month of 20 working days is extremely rare.

    Returns a boolean Series aligned to df.index (True = flagged).
    """
    if "avg_hours_worked" not in df.columns:
        return pd.Series(False, index=df.index)

    return (df["avg_hours_worked"] > max_plausible_hours).rename("rule_impossible_hours")


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all business rules to the raw attendance DataFrame.

    Returns a copy of df with two new boolean columns:
        ``rule_zero_attendance``: Rule 1 (see above)
        ``rule_impossible_hours``: Rule 2 (see above)
    """
    out = df.copy()
    out["rule_zero_attendance"] = rule_zero_attendance(df).astype(int)
    out["rule_impossible_hours"] = rule_impossible_hours(df).astype(int)
    return out


# ---------------------------------------------------------------------------
# Combining model and rules
# ---------------------------------------------------------------------------
def combine_flags(model_flag: np.ndarray,
                  rule_zero: pd.Series,
                  rule_hours: pd.Series) -> pd.Series:
    """combined_anomaly = model_anomaly OR rule_zero_attendance OR rule_impossible_hours.

    Adding rules improves recall (catches more true anomalies) at the cost of
    lower precision (more false positives).  Each source is kept separately so
    the HR team can see why a record was flagged.

    Returns a boolean (int) Series.
    """
    combined = (
        (model_flag == 1)
        | rule_zero.astype(bool)
        | rule_hours.astype(bool)
    ).astype(int)
    return combined.rename("combined_anomaly")


def build_reason(row: pd.Series) -> str:
    """Human-readable explanation of why a record was flagged."""
    reasons = []
    if row.get("model_anomaly", 0):
        reasons.append("IsolationForest")
    if row.get("rule_zero_attendance", 0):
        reasons.append("zero-attendance rule")
    if row.get("rule_impossible_hours", 0):
        reasons.append("impossible-hours rule")
    return "; ".join(reasons) if reasons else ""


# ---------------------------------------------------------------------------
# HR priority ranking
# ---------------------------------------------------------------------------
def priority_rank(results: pd.DataFrame) -> pd.DataFrame:
    """Add a ``priority`` column (High / Medium / Low) and sort flagged records.

    Ranking factors (all apply only within flagged records):
    1. ``anomaly_score`` (from IsolationForest): higher = more isolated = more suspicious.
    2. ``rule_count``: how many independent anomaly signals agree (0, 1, 2, or 3).
    3. The two rules are weighted: impossible_hours > zero_attendance because
       IMPOSSIBLE_HOURS anomalies are clear data errors with financial impact.

    The composite score is:
        priority_score = anomaly_score + 0.5 * rule_impossible_hours + 0.3 * rule_zero_attendance

    Thresholds (applied only within the flagged subset):
    - Top one-third by priority_score → High
    - Middle third → Medium
    - Bottom third → Low

    Non-flagged records get priority = "Not flagged".
    """
    df = results.copy()

    # Composite score
    df["priority_score"] = (
        df["anomaly_score"]
        + 0.5 * df.get("rule_impossible_hours", 0).fillna(0)
        + 0.3 * df.get("rule_zero_attendance", 0).fillna(0)
    )

    # Assign priority only within flagged records
    flagged_mask = df["combined_anomaly"] == 1
    df["priority"] = "Not flagged"

    if flagged_mask.sum() > 0:
        flagged_scores = df.loc[flagged_mask, "priority_score"]
        tertile_lo = flagged_scores.quantile(1 / 3)
        tertile_hi = flagged_scores.quantile(2 / 3)

        priority_series = pd.Series("Low", index=flagged_scores.index)
        priority_series[flagged_scores >= tertile_hi] = "High"
        priority_series[(flagged_scores >= tertile_lo) & (flagged_scores < tertile_hi)] = "Medium"

        df.loc[flagged_mask, "priority"] = priority_series

    # Sort: flagged first, then by priority_score descending
    priority_order = {"High": 0, "Medium": 1, "Low": 2, "Not flagged": 3}
    df["_priority_order"] = df["priority"].map(priority_order)
    df = df.sort_values(["_priority_order", "priority_score"], ascending=[True, False])
    df = df.drop(columns=["_priority_order"])

    return df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def catch_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of true anomalies that the detector flagged (= recall).

    This is the primary metric referenced in the task brief as ``catch_rate()``.
    The anomaly baseline shows recall = 0.593 for the reference implementation.

    Parameters
    ----------
    y_true : 1-D array-like, dtype int (1 = true anomaly, 0 = normal)
    y_pred : 1-D array-like, dtype int (1 = flagged, 0 = not flagged)

    Returns
    -------
    float in [0, 1]
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.sum() == 0:
        raise ValueError("y_true has no positive labels; recall is undefined.")
    return float(recall_score(yt, yp, zero_division=0))


def evaluate(results: pd.DataFrame, path=WORKBOOK) -> dict:
    """Join detector predictions with hidden validation labels and compute metrics.

    The labels are loaded HERE (after predictions are ready) and are never used
    earlier in the pipeline.

    Parameters
    ----------
    results : the DataFrame produced by ``run_detector()`` (must contain
              ``attendance_feature_id`` and ``combined_anomaly``).
    path    : path to the master workbook.

    Returns
    -------
    dict with keys: precision, recall, f1, true_positives, false_positives,
                    false_negatives, flagged, injected, flagged_pct, flagged_pct_of_all
    """
    labels = load_labels(path)
    # Create a binary label column for every record in `results`
    labelled_ids = set(labels["attendance_feature_id"])
    results = results.copy()
    results["is_true_anomaly"] = results["attendance_feature_id"].isin(labelled_ids).astype(int)

    y_true = results["is_true_anomaly"]
    y_pred = results["combined_anomaly"]

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "flagged": int(y_pred.sum()),
        "injected": int(y_true.sum()),
        "flagged_pct_of_all": float(y_pred.mean()),
        "catch_rate": catch_rate(y_true, y_pred),
    }


def evaluate_model_only(results: pd.DataFrame, path=WORKBOOK) -> dict:
    """Same as evaluate() but uses only the IsolationForest flag (no rules)."""
    labels = load_labels(path)
    labelled_ids = set(labels["attendance_feature_id"])
    results = results.copy()
    results["is_true_anomaly"] = results["attendance_feature_id"].isin(labelled_ids).astype(int)
    y_true = results["is_true_anomaly"]
    y_pred = results["model_anomaly"]
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "flagged": int(y_pred.sum()),
        "injected": int(y_true.sum()),
        "flagged_pct_of_all": float(y_pred.mean()),
        "catch_rate": catch_rate(y_true, y_pred),
    }


def evaluate_rules_only(results: pd.DataFrame, path=WORKBOOK) -> dict:
    """Evaluate just the union of business rules (without IsolationForest)."""
    labels = load_labels(path)
    labelled_ids = set(labels["attendance_feature_id"])
    results = results.copy()
    results["is_true_anomaly"] = results["attendance_feature_id"].isin(labelled_ids).astype(int)
    y_true = results["is_true_anomaly"]
    # Union of rule flags
    y_pred = ((results["rule_zero_attendance"] == 1) | (results["rule_impossible_hours"] == 1)).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "flagged": int(y_pred.sum()),
        "injected": int(y_true.sum()),
        "flagged_pct_of_all": float(y_pred.mean()),
        "catch_rate": catch_rate(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Missed anomaly analysis
# ---------------------------------------------------------------------------
def analyse_misses(results: pd.DataFrame, path=WORKBOOK) -> pd.DataFrame:
    """Return the false-negative records joined with their anomaly type.

    Call this only after predictions are ready.  Uses hidden labels for error
    analysis only, never for training.
    """
    labels = load_labels(path)
    labelled_ids = set(labels["attendance_feature_id"])
    results = results.copy()
    results["is_true_anomaly"] = results["attendance_feature_id"].isin(labelled_ids).astype(int)

    fn_mask = (results["is_true_anomaly"] == 1) & (results["combined_anomaly"] == 0)
    fn_records = results[fn_mask].copy()

    # Join with anomaly type
    fn_with_type = fn_records.merge(
        labels[["attendance_feature_id", "anomaly_type", "injection_note"]],
        on="attendance_feature_id", how="left"
    )
    return fn_with_type


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_detector(path=WORKBOOK,
                 features: Sequence[str] = ATTENDANCE_FEATURES,
                 contamination: float = CONTAMINATION,
                 random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """End-to-end attendance anomaly detection pipeline.

    Steps
    -----
    1. Load attendance data.
    2. Select numeric features.
    3. Handle missing/infinite values.
    4. Drop constant columns.
    5. Scale features (StandardScaler fitted on the full attendance table).
    6. Fit IsolationForest and extract flags + scores.
    7. Apply business rules on the raw data.
    8. Combine into a single anomaly flag.
    9. Build reason strings.
    10. Rank flagged records.

    Returns
    -------
    DataFrame with one row per attendance record, plus columns:
        model_anomaly, anomaly_score, rule_zero_attendance, rule_impossible_hours,
        combined_anomaly, anomaly_reason, priority_score, priority
    """
    # 1. Load
    att = load_attendance(path)
    print(f"Loaded {len(att):,} attendance records for {att['employee_id'].nunique()} employees.")

    # 2. Select features
    X = select_features(att, features)

    # 3. Missing values
    X_clean = handle_missing(X)

    # 4. Drop constant columns
    X_clean, dropped = drop_constant_columns(X_clean)
    if dropped:
        print(f"  Dropped constant columns: {dropped}")

    # 5. Scale
    scaler = fit_scaler(X_clean)
    X_scaled = scaler.transform(X_clean)

    # 6. Isolation Forest
    model = build_model(contamination=contamination, random_state=random_state)
    model_flag, score = fit_predict_model(model, X_scaled)

    # 7. Business rules (applied to raw data, not scaled data)
    att_with_rules = apply_rules(att)

    # 8. Combine
    combined = combine_flags(model_flag,
                             att_with_rules["rule_zero_attendance"],
                             att_with_rules["rule_impossible_hours"])

    # 9. Assemble results
    results = att[["attendance_feature_id", "employee_id", "month",
                   "department_id"] + list(features)].copy()
    results["model_anomaly"] = model_flag
    results["anomaly_score"] = score
    results["rule_zero_attendance"] = att_with_rules["rule_zero_attendance"].values
    results["rule_impossible_hours"] = att_with_rules["rule_impossible_hours"].values
    results["combined_anomaly"] = combined.values
    results["anomaly_reason"] = results.apply(build_reason, axis=1)

    # 10. Rank
    results = priority_rank(results)

    n_flagged = results["combined_anomaly"].sum()
    n_model = results["model_anomaly"].sum()
    n_r1 = results["rule_zero_attendance"].sum()
    n_r2 = results["rule_impossible_hours"].sum()
    print(f"Flagged: {n_flagged} total  "
          f"(model {n_model}, zero-attendance rule {n_r1}, impossible-hours rule {n_r2})")

    return results

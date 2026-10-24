"""Unit tests for attendance anomaly detection (Task 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cortexhr.attendance_anomaly import (
    ATTENDANCE_FEATURES,
    CONTAMINATION,
    apply_rules,
    build_model,
    build_reason,
    catch_rate,
    combine_flags,
    drop_constant_columns,
    fit_predict_model,
    fit_scaler,
    handle_missing,
    priority_rank,
    rule_impossible_hours,
    rule_zero_attendance,
    select_features,
)


@pytest.fixture
def sample_attendance_df() -> pd.DataFrame:
    """Synthetic attendance data mimicking attendance_monthly_features."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "attendance_feature_id": [f"AF-{i:05d}" for i in range(n)],
        "employee_id": [f"EMP-{(i % 20):04d}" for i in range(n)],
        "month": ["2025-01-01"] * n,
        "department_id": ["DEP-001"] * n,
        "scheduled_days": [20.0] * n,
        "present_days": np.random.uniform(15, 20, size=n),
        "half_days_unpaid": [0.0] * n,
        "paid_leave_days": np.random.uniform(0, 2, size=n),
        "unpaid_absence_days": np.random.uniform(0, 1, size=n),
        "longest_unpaid_streak_days": [0.0] * n,
        "late_arrivals": np.random.poisson(0.5, size=n).astype(float),
        "avg_hours_worked": np.random.normal(8.0, 0.5, size=n),
        "overtime_hours": np.random.exponential(2.0, size=n),
        "attendance_rate": np.random.uniform(0.85, 1.0, size=n),
    })

    # Inject intentional anomalies for rule testing
    # Row 0: zero attendance without full paid leave
    df.loc[0, "present_days"] = 0.0
    df.loc[0, "paid_leave_days"] = 0.0
    df.loc[0, "attendance_rate"] = 0.0

    # Row 1: zero attendance with full paid leave (should NOT be flagged by rule 1)
    df.loc[1, "present_days"] = 0.0
    df.loc[1, "paid_leave_days"] = 20.0
    df.loc[1, "attendance_rate"] = 0.0

    # Row 2: impossible hours (e.g. 18.5 hours/day)
    df.loc[2, "avg_hours_worked"] = 18.5

    return df


def test_select_features_valid(sample_attendance_df):
    X = select_features(sample_attendance_df)
    assert list(X.columns) == ATTENDANCE_FEATURES
    assert len(X) == len(sample_attendance_df)
    assert not any(c in X.columns for c in ["employee_id", "attendance_feature_id", "month"])


def test_select_features_missing_col_raises(sample_attendance_df):
    incomplete_df = sample_attendance_df.drop(columns=["present_days"])
    with pytest.raises(KeyError, match="present_days"):
        select_features(incomplete_df)


def test_handle_missing_and_inf():
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, np.inf],
        "b": [10.0, 20.0, -np.inf, 40.0],
    })
    clean = handle_missing(df)
    assert not clean.isna().any().any()
    assert not np.isinf(clean.to_numpy()).any()
    # Check that median was filled for nan/inf
    assert clean.loc[1, "a"] == 2.0  # median of [1.0, 3.0]


def test_drop_constant_columns():
    df = pd.DataFrame({
        "varying": [1.0, 2.0, 3.0],
        "constant": [5.0, 5.0, 5.0],
    })
    filtered, dropped = drop_constant_columns(df)
    assert "constant" in dropped
    assert "constant" not in filtered.columns
    assert "varying" in filtered.columns


def test_rule_zero_attendance(sample_attendance_df):
    rule = rule_zero_attendance(sample_attendance_df)
    # Row 0: 0 present, 0 paid leave -> True
    assert rule.iloc[0] == True
    # Row 1: 0 present, 20 paid leave -> False (exempt)
    assert rule.iloc[1] == False
    # Row 3: normal -> False
    assert rule.iloc[3] == False


def test_rule_impossible_hours(sample_attendance_df):
    rule = rule_impossible_hours(sample_attendance_df, max_plausible_hours=14.0)
    # Row 2 has 18.5 -> True
    assert rule.iloc[2] == True
    # Row 3 has normal (~8) -> False
    assert rule.iloc[3] == False


def test_apply_rules(sample_attendance_df):
    df_rules = apply_rules(sample_attendance_df)
    assert "rule_zero_attendance" in df_rules.columns
    assert "rule_impossible_hours" in df_rules.columns
    assert df_rules.loc[0, "rule_zero_attendance"] == 1
    assert df_rules.loc[2, "rule_impossible_hours"] == 1


def test_combine_flags():
    model_flag = np.array([1, 0, 0, 0])
    rule_zero = pd.Series([0, 1, 0, 0])
    rule_hours = pd.Series([0, 0, 1, 0])
    combined = combine_flags(model_flag, rule_zero, rule_hours)
    expected = pd.Series([1, 1, 1, 0], name="combined_anomaly")
    pd.testing.assert_series_equal(combined, expected)


def test_priority_rank():
    results = pd.DataFrame({
        "attendance_feature_id": ["A1", "A2", "A3", "A4"],
        "anomaly_score": [0.8, 0.4, 0.1, 0.05],
        "rule_impossible_hours": [1, 0, 0, 0],
        "rule_zero_attendance": [0, 1, 0, 0],
        "combined_anomaly": [1, 1, 1, 0],
    })
    ranked = priority_rank(results)
    assert "priority" in ranked.columns
    assert "priority_score" in ranked.columns
    # Non-flagged record gets "Not flagged"
    assert ranked.loc[ranked["attendance_feature_id"] == "A4", "priority"].iloc[0] == "Not flagged"
    # Highest score gets High
    assert ranked.loc[ranked["attendance_feature_id"] == "A1", "priority"].iloc[0] == "High"


def test_catch_rate():
    y_true = pd.Series([1, 1, 1, 1, 0, 0])
    y_pred = pd.Series([1, 1, 0, 0, 1, 0])
    # 2 TP out of 4 injected = 0.50
    rate = catch_rate(y_true, y_pred)
    assert rate == pytest.approx(0.50)


def test_isolation_forest_unsupervised_training(sample_attendance_df):
    """Ensure Isolation Forest trains and predicts with correct contamination and shapes."""
    X = select_features(sample_attendance_df)
    X_clean = handle_missing(X)
    X_clean, _ = drop_constant_columns(X_clean)
    scaler = fit_scaler(X_clean)
    X_scaled = scaler.transform(X_clean)

    model = build_model(contamination=0.05, random_state=42)
    flags, scores = fit_predict_model(model, X_scaled)

    assert len(flags) == len(sample_attendance_df)
    assert len(scores) == len(sample_attendance_df)
    assert set(np.unique(flags)).issubset({0, 1})
    # Roughly 5% contamination on 100 samples ~ 5 anomalies
    assert 3 <= flags.sum() <= 8

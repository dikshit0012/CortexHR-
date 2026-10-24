#!/usr/bin/env python
"""Day 5 / Task 5: Run Attendance Anomaly Detection and Evaluation.

Usage:
    python scripts/run_attendance_anomaly.py
    python scripts/run_attendance_anomaly.py --contamination 0.03

Steps executed:
1. Load attendance_monthly_features from the workbook.
2. Preprocess 10 numeric attendance features (median imputation, scaling).
3. Train Isolation Forest (contamination ~3%, n_estimators=200, random_state=42) without labels.
4. Apply 2 business rules (zero attendance with scheduled days, impossible hours worked > 14h/day).
5. Combine model + rules with transparent OR logic and reason tagging.
6. Rank flagged anomalies by priority score into High, Medium, and Low tiers.
7. Evaluate against hidden validation labels in attendance_anomaly_labels sheet using catch_rate() and precision.
8. Compare with reference baseline from anomaly_baselines sheet.
9. Investigate false negatives across all 5 injected anomaly types.
10. Export results, metrics tables, and findings JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cortexhr.attendance_anomaly import (
    ATTENDANCE_FEATURES,
    CONTAMINATION,
    RANDOM_STATE,
    analyse_misses,
    evaluate,
    evaluate_model_only,
    evaluate_rules_only,
    load_anomaly_baselines,
    load_attendance,
    load_labels,
    run_detector,
)
from cortexhr.paths import DATA_PROCESSED, TABLES, WORKBOOK


def main(contamination: float = CONTAMINATION, seed: int = RANDOM_STATE) -> dict:
    print("=" * 70)
    print("CortexHR Day 5 / Task 5: Attendance Anomaly Detection Pipeline")
    print("=" * 70)

    if not WORKBOOK.exists():
        sys.exit(f"Missing master dataset workbook at {WORKBOOK}")

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # 1. Run Detector (Unsupervised training + rules)
    print("\n[Step 1/5] Running Unsupervised Anomaly Detection...")
    results = run_detector(path=WORKBOOK, contamination=contamination, random_state=seed)

    # 2. Evaluate against Hidden Validation Labels
    print("\n[Step 2/5] Evaluating against Validation Vault Labels...")
    eval_model = evaluate_model_only(results, path=WORKBOOK)
    eval_rules = evaluate_rules_only(results, path=WORKBOOK)
    eval_combined = evaluate(results, path=WORKBOOK)

    # Load baseline from workbook
    baseline_df = load_anomaly_baselines(WORKBOOK)
    att_baseline = baseline_df[baseline_df["task"].str.contains("Attendance", case=False, na=False)].iloc[0]

    # Comparison Table
    comparison_rows = [
        {
            "approach": "Workbook Baseline (Ref Isolation Forest)",
            "flagged": int(att_baseline["flagged"]),
            "injected": int(att_baseline["injected"]),
            "precision": float(att_baseline["precision"]),
            "recall_catch_rate": float(att_baseline["recall"]),
            "f1_score": round(2 * (att_baseline["precision"] * att_baseline["recall"]) / (att_baseline["precision"] + att_baseline["recall"]), 4),
            "tp": int(round(att_baseline["recall"] * att_baseline["injected"])),
            "fp": int(att_baseline["flagged"] - round(att_baseline["recall"] * att_baseline["injected"])),
            "fn": int(att_baseline["injected"] - round(att_baseline["recall"] * att_baseline["injected"])),
        },
        {
            "approach": "Our Isolation Forest (Model Only)",
            "flagged": eval_model["flagged"],
            "injected": eval_model["injected"],
            "precision": round(eval_model["precision"], 4),
            "recall_catch_rate": round(eval_model["recall"], 4),
            "f1_score": round(eval_model["f1"], 4),
            "tp": eval_model["true_positives"],
            "fp": eval_model["false_positives"],
            "fn": eval_model["false_negatives"],
        },
        {
            "approach": "Business Rules Only (Zero Attendance + Impossible Hours)",
            "flagged": eval_rules["flagged"],
            "injected": eval_rules["injected"],
            "precision": round(eval_rules["precision"], 4),
            "recall_catch_rate": round(eval_rules["recall"], 4),
            "f1_score": round(eval_rules["f1"], 4),
            "tp": eval_rules["true_positives"],
            "fp": eval_rules["false_positives"],
            "fn": eval_rules["false_negatives"],
        },
        {
            "approach": "Our Combined Pipeline (Model OR Business Rules)",
            "flagged": eval_combined["flagged"],
            "injected": eval_combined["injected"],
            "precision": round(eval_combined["precision"], 4),
            "recall_catch_rate": round(eval_combined["recall"], 4),
            "f1_score": round(eval_combined["f1"], 4),
            "tp": eval_combined["true_positives"],
            "fp": eval_combined["false_positives"],
            "fn": eval_combined["false_negatives"],
        },
    ]
    comparison_table = pd.DataFrame(comparison_rows)
    print("\n=== PERFORMANCE COMPARISON ===")
    print(comparison_table.to_string(index=False))

    # 3. Analyze Misses (False Negatives by Anomaly Type)
    print("\n[Step 3/5] Analyzing Missed Anomalies (False Negatives)...")
    misses = analyse_misses(results, path=WORKBOOK)
    labels = load_labels(WORKBOOK)
    type_counts = labels["anomaly_type"].value_counts().rename("injected_count")
    miss_counts = misses["anomaly_type"].value_counts().rename("missed_count")

    type_analysis = pd.concat([type_counts, miss_counts], axis=1).fillna(0).astype(int)
    type_analysis["caught_count"] = type_analysis["injected_count"] - type_analysis["missed_count"]
    type_analysis["catch_rate_pct"] = (type_analysis["caught_count"] / type_analysis["injected_count"] * 100).round(1)
    type_analysis = type_analysis.reset_index().rename(columns={"index": "anomaly_type"})

    print("\n=== ANOMALY CATCH RATE BY INJECTED TYPE ===")
    print(type_analysis.to_string(index=False))

    # 4. Priority Breakdown for HR Review
    print("\n[Step 4/5] Computing HR Review Priority Summary...")
    flagged_mask = results["combined_anomaly"] == 1
    flagged_df = results[flagged_mask].copy()

    # Join labels to check TP precision per priority band
    labelled_ids = set(labels["attendance_feature_id"])
    flagged_df["is_true_anomaly"] = flagged_df["attendance_feature_id"].isin(labelled_ids).astype(int)

    priority_summary = flagged_df.groupby("priority").agg(
        total_flagged=("attendance_feature_id", "count"),
        true_anomalies=("is_true_anomaly", "sum"),
        min_priority_score=("priority_score", "min"),
        mean_priority_score=("priority_score", "mean"),
        max_priority_score=("priority_score", "max"),
    ).reset_index()
    priority_summary["precision_pct"] = (priority_summary["true_anomalies"] / priority_summary["total_flagged"] * 100).round(1)
    # Order High, Medium, Low
    order_dict = {"High": 0, "Medium": 1, "Low": 2}
    priority_summary["order"] = priority_summary["priority"].map(order_dict)
    priority_summary = priority_summary.sort_values("order").drop(columns=["order"])

    print("\n=== HR REVIEW PRIORITY BANDS ===")
    print(priority_summary.to_string(index=False))

    # 5. Format to Match sample_anomaly_results Schema
    print("\n[Step 5/5] Exporting Deliverables and Evidence...")
    # Schema: result_id, source_dataset, record_id, employee_id, period, anomaly_score, flagged, reason_code, reason_text, money_at_risk, model_version, scored_at
    scored_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def map_reason_code(row):
        codes = []
        if row["rule_impossible_hours"]:
            codes.append("IMPOSSIBLE_HOURS")
        if row["rule_zero_attendance"]:
            codes.append("NO_ATTENDANCE")
        if row["model_anomaly"]:
            codes.append("IFOREST_OUTLIER")
        return ";".join(codes) if codes else "NORMAL"

    sample_format_records = []
    for idx, (_, row) in enumerate(flagged_df.iterrows(), start=1):
        sample_format_records.append({
            "result_id": f"AN-{idx:04d}",
            "source_dataset": "attendance_monthly_features",
            "record_id": row["attendance_feature_id"],
            "employee_id": row["employee_id"],
            "period": str(row["month"])[:10],
            "anomaly_score": round(float(row["priority_score"]), 4),
            "flagged": "Y",
            "reason_code": map_reason_code(row),
            "reason_text": row["anomaly_reason"],
            "money_at_risk": np.nan,
            "model_version": "iforest-attendance-v1.0-rules-hybrid",
            "scored_at": scored_time,
        })
    df_sample_format = pd.DataFrame(sample_format_records)

    # Save Tables and Files
    comparison_table.to_csv(TABLES / "day5_baseline_comparison.csv", index=False)
    type_analysis.to_csv(TABLES / "day5_missed_anomalies_by_type.csv", index=False)
    priority_summary.to_csv(TABLES / "day5_priority_summary.csv", index=False)
    comparison_table.to_json(TABLES / "day5_anomaly_metrics.json", orient="records", indent=2)

    results.to_csv(DATA_PROCESSED / "attendance_anomaly_all_scored.csv", index=False)
    flagged_df.to_csv(DATA_PROCESSED / "attendance_anomaly_flagged.csv", index=False)
    df_sample_format.to_csv(DATA_PROCESSED / "sample_format_anomaly_results.csv", index=False)

    findings = {
        "timestamp": scored_time,
        "contamination": contamination,
        "features": ATTENDANCE_FEATURES,
        "total_records": len(results),
        "total_flagged": int(eval_combined["flagged"]),
        "injected_labels": int(eval_combined["injected"]),
        "metrics": {
            "model_only": eval_model,
            "rules_only": eval_rules,
            "combined": eval_combined,
        },
        "baseline_comparison": {
            "baseline_precision": float(att_baseline["precision"]),
            "baseline_recall": float(att_baseline["recall"]),
            "our_precision": round(eval_combined["precision"], 4),
            "our_recall": round(eval_combined["recall"], 4),
            "precision_gain": round(eval_combined["precision"] - float(att_baseline["precision"]), 4),
            "recall_gain": round(eval_combined["recall"] - float(att_baseline["recall"]), 4),
        },
        "missed_anomalies": {
            "total_missed": int(eval_combined["false_negatives"]),
            "by_type": type_analysis.to_dict(orient="records"),
            "root_cause": (
                "Isolation Forest operates in 10-dimensional space where LATE_SPIKE records "
                "exhibit 9 completely normal feature values (100% attendance rate, 0 unpaid absences, "
                "normal working hours). Unsupervised random partition cuts rarely isolate such records "
                "early without a specific business rule on late arrivals."
            ),
        },
        "priority_bands": priority_summary.to_dict(orient="records"),
    }
    with open(TABLES / "day5_anomaly_findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

    print("\n[Done] All Task 5 artifacts and metrics written successfully!")
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 5 Attendance Anomaly Detector")
    parser.add_argument("--contamination", type=float, default=CONTAMINATION, help="Isolation Forest contamination rate")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    args = parser.parse_args()
    main(contamination=args.contamination, seed=args.seed)

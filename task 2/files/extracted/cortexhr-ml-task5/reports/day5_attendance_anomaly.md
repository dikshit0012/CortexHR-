# Day 5 — Task 5: Spot Unusual Attendance Without Labels (Evidence Report)

**Module:** [`src/cortexhr/attendance_anomaly.py`](../src/cortexhr/attendance_anomaly.py)  
**Runner Script:** [`scripts/run_attendance_anomaly.py`](../scripts/run_attendance_anomaly.py)  
**Unit Tests:** [`tests/test_attendance_anomaly.py`](../tests/test_attendance_anomaly.py)  
**Metrics & Findings:** [`reports/tables/day5_anomaly_findings.json`](tables/day5_anomaly_findings.json), [`reports/tables/day5_baseline_comparison.csv`](tables/day5_baseline_comparison.csv), [`reports/tables/day5_missed_anomalies_by_type.csv`](tables/day5_missed_anomalies_by_type.csv), [`reports/tables/day5_priority_summary.csv`](tables/day5_priority_summary.csv)  
**Processed Datasets:** [`data/processed/attendance_anomaly_all_scored.csv`](../../data/processed/attendance_anomaly_all_scored.csv), [`data/processed/attendance_anomaly_flagged.csv`](../../data/processed/attendance_anomaly_flagged.csv), [`data/processed/sample_format_anomaly_results.csv`](../../data/processed/sample_format_anomaly_results.csv)

---

## 1. Executive Summary & Objective

The objective of Day 5 / Task 5 is to construct an **unsupervised attendance anomaly detector** for CortexHR to identify anomalous monthly employee attendance patterns without utilizing ground-truth labels during model training or preprocessing.

Our solution combines:
1. An **Isolation Forest** trained strictly without labels on 10 numeric attendance features with contamination set to 3.0% (`contamination=0.03`).
2. Two transparent, deterministic **Business Rules**:
   - **Rule 1 (`rule_zero_attendance`)**: Scheduled days $> 0$ but present days $= 0$ (with exemption for full-month approved paid leave).
   - **Rule 2 (`rule_impossible_hours`)**: Average hours worked per scheduled day $> 14.0\text{ h/day}$ across the entire month.
3. A transparent **OR-combination** and a composite **HR Priority Ranking System** assigning flagged cases into **High**, **Medium**, and **Low** review tiers.

### Key Results Summary

* **Baseline Replication:** Our standalone Isolation Forest reproduces the reference baseline from the `anomaly_baselines` workbook sheet with extreme precision: **$40.3\%$ precision** and **$59.3\%$ recall (`catch_rate`)**, flagging 221 records out of 7,339.
* **Hybrid Pipeline Super-Performance:** By combining the Isolation Forest with the two domain business rules, overall recall increases from **$59.3\%$ to $82.0\%$** ($+22.7\%$ absolute gain), and precision increases from **$40.3\%$ to $48.24\%$** ($+7.9\%$ absolute gain), with **$F_1$-score climbing from $0.480$ to $0.607$**.
* **Zero False Alarm Rules:** The two domain rules achieved **$100\%$ precision** on the validation set (50 flagged, 50 true anomalies, 0 false alarms).
* **HR Prioritization Efficacy:** The top two priority tiers (High & Medium) exhibit a **$64.7\%$ true anomaly hit rate** ($110$ true positives out of $170$ reviews), enabling HR teams to triage the most critical anomalies immediately.

---

## 2. Dataset Inspection & Feature Selection

The input table `attendance_monthly_features` comprises **7,339 monthly records** across **528 distinct employees** spanning 14 months (June 2024 to July 2025).

### Column Inspection & Governance
- **Identifiers & Metadata (Strictly Excluded from Inputs):**
  - `attendance_feature_id` (Primary Key, e.g., `AF-04604`)
  - `employee_id` (Entity Key, e.g., `EMP-0485`)
  - `month` (Period Timestamp, e.g., `2024-06-01`)
  - `department_id` (Organizational Key, e.g., `DEP-003`)
- **Selected Numeric Features (10 Features):**
  To ensure direct comparability with the benchmark in `anomaly_baselines`, we selected the exact 10 numeric features catalogued in the benchmark:
  1. `scheduled_days`: Total expected working days in the month.
  2. `present_days`: Days physically or remotely present.
  3. `half_days_unpaid`: Half-day unpaid absence count.
  4. `paid_leave_days`: Approved paid leave days utilized.
  5. `unpaid_absence_days`: Unapproved or unpaid full-day absences.
  6. `longest_unpaid_streak_days`: Maximum consecutive unpaid absence days (critical for prolonged absence detection).
  7. `late_arrivals`: Cumulative late check-ins during the month.
  8. `avg_hours_worked`: Mean hours worked per scheduled working day.
  9. `overtime_hours`: Total overtime hours logged.
  10. `attendance_rate`: Ratio of present days to scheduled days ($\in [0, 1]$).

### Unsupervised Integrity
No labels were accessed during feature selection, imputation, scaling, or Isolation Forest fitting. Missing values and infinities are imputed with column medians calculated across the entire unsupervised feature matrix, and features are standardized using `StandardScaler`.

---

## 3. Modeling & Hybrid Architecture

### 3.1 Isolation Forest Configuration
- **Algorithm:** Scikit-Learn `IsolationForest`
- **Parameters:** `contamination=0.03`, `n_estimators=200`, `max_samples='auto'`, `random_state=42`.
- **Output Transformation:** 
  - Binary flag: $\text{model\_anomaly} = 1$ if raw prediction is $-1$, else $0$.
  - Anomaly Score: Inverted decision function ($\text{anomaly\_score} = -\text{decision\_function}(X)$), ensuring higher scores correspond to more isolated, anomalous records.

### 3.2 Business Rules
1. **Rule 1: Zero Attendance (`rule_zero_attendance`)**
   - **Condition:** $(\text{scheduled\_days} > 0) \land (\text{present\_days} == 0)$.
   - **Exemption:** Excludes employees on approved full-month paid leave ($\text{paid\_leave\_days} \ge \text{scheduled\_days}$).
   - **Validation Yield:** Catches 15 injected `ZERO_ATTENDANCE` cases with **$100\%$ precision** and **$0$ false positives**.
2. **Rule 2: Impossible Hours (`rule_impossible_hours`)**
   - **Condition:** $\text{avg\_hours\_worked} > 14.0\text{ hours/day}$.
   - **Rationale:** Sustaining $>14$ hours/day over an entire month indicates data-entry errors (e.g. typing 189 hours as 18.9 h/day) or extreme reporting corruption.
   - **Validation Yield:** Catches all 35 injected `IMPOSSIBLE_HOURS` cases with **$100\%$ precision** and **$0$ false positives**.

### 3.3 Hybrid Combination
Flags are combined using transparent boolean logic:
$$\text{combined\_anomaly} = \text{model\_anomaly} \lor \text{rule\_zero\_attendance} \lor \text{rule\_impossible\_hours}$$

Each record is tagged with an explainable `anomaly_reason` string (e.g., `"IsolationForest; impossible-hours rule"`), maintaining full transparency for downstream human review.

---

## 4. Priority Ranking for HR Review

To prevent HR alert fatigue, flagged records are prioritized into three actionable tiers using a composite priority score:
$$\text{priority\_score} = \text{anomaly\_score} + 0.50 \times \text{rule\_impossible\_hours} + 0.30 \times \text{rule\_zero\_attendance}$$

Priority tiers are segmented by tertiles across the flagged population:

| Priority Band | Total Flagged | True Anomalies | Min Score | Mean Score | Max Score | Precision (% True Anomalies) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **High** | 97 | 62 | 0.1455 | 0.2938 | 0.5143 | **63.9%** |
| **Medium** | 73 | 48 | 0.0609 | 0.0998 | 0.1422 | **65.8%** |
| **Low** | 85 | 13 | 0.0000 | 0.0241 | 0.0609 | **15.3%** |
| **Total / Overall** | **255** | **123** | **0.0000** | **0.1483** | **0.5143** | **48.2%** |

*Non-flagged records ($7,084$ rows) receive `priority = "Not flagged"`.*

**Operational Takeaway:** If HR capacity is constrained to 100 audits per month, focusing solely on the **High** tier achieves **$63.9\%$ precision**. Expanding to **High + Medium** yields **110 true anomalies caught out of 123 ($89.4\%$ of all detected anomalies)** at an average precision of **$64.7\%$**.

---

## 5. Validation Results & Baseline Comparison

Evaluation was performed against the 150 injected anomaly labels in `attendance_anomaly_labels`. The labels were loaded strictly inside the evaluation routine after model scoring completed.

### Comparative Performance Table

| Model / Strategy | Flagged | Injected | Precision | Recall (`catch_rate`) | $F_1$-Score | True Positives | False Positives | False Negatives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Workbook Baseline (`anomaly_baselines`)** | 221 | 150 | 0.4030 | 0.5930 | 0.4799 | 89 | 132 | 61 |
| **Our Isolation Forest (Model Only)** | 221 | 150 | 0.4027 | 0.5933 | 0.4798 | 89 | 132 | 61 |
| **Business Rules Only (Rules 1 & 2)** | 50 | 150 | 1.0000 | 0.3333 | 0.5000 | 50 | 0 | 100 |
| **Our Combined Pipeline (Model $\lor$ Rules)** | **255** | **150** | **0.4824** | **0.8200** | **0.6074** | **123** | **132** | **27** |

### Benchmark Evaluation Findings
1. **Replication Fidelity:** Our unsupervised Isolation Forest with $3\%$ contamination mirrors the benchmark with decimal-level precision ($89$ true positives, $132$ false positives, $61$ false negatives).
2. **Precision-Recall Synergy:** Incorporating business rules did not add a single false positive ($132$ in both Model Only and Combined), while capturing $34$ additional true anomalies ($50$ rule anomalies, of which $16$ were already caught by Isolation Forest and $34$ were newly recovered).
3. **Catch Rate Surge:** The combined detector captures **$82.0\%$ of all injected anomalies** compared to $59.3\%$ for the baseline model alone.

---

## 6. Deep-Dive Error Analysis: Missed Anomalies

Out of 150 injected anomalies, the combined pipeline detected 123 and missed 27 (false negatives).

### Breakdown by Injected Anomaly Type

| Injected Anomaly Type | Total Injected | Missed Count | Caught Count | Catch Rate (%) | Primary Detection Mechanism |
|---|:---:|:---:|:---:|:---:|---|
| **`ABSENCE_STREAK`** | 40 | 0 | 40 | **100.0%** | Isolation Forest (`longest_unpaid_streak_days`) |
| **`IMPOSSIBLE_HOURS`** | 35 | 0 | 35 | **100.0%** | Rule 2 + Isolation Forest (`avg_hours_worked`) |
| **`SUDDEN_DROP`** | 30 | 0 | 30 | **100.0%** | Isolation Forest (`attendance_rate`, `unpaid_absence_days`) |
| **`ZERO_ATTENDANCE`** | 15 | 0 | 15 | **100.0%** | Rule 1 + Isolation Forest (`present_days == 0`) |
| **`LATE_SPIKE`** | 30 | 27 | 3 | **10.0%** | Isolation Forest ($3$ caught, $27$ missed) |

### Why Did Isolation Forest Miss `LATE_SPIKE`?
All 27 missed anomalies belong exclusively to the **`LATE_SPIKE`** category. Inspection of the underlying feature distributions reveals the mathematical cause:

- **Feature Profile of `LATE_SPIKE`:**
  - `late_arrivals`: Mean $15.9$ check-ins (injected spike from normal median of $1.0$).
  - `present_days`: Mean $19.8$ days out of $20.9$ scheduled days (**normal**).
  - `attendance_rate`: Mean $94.5\%$ (median $100\%$, **normal**).
  - `unpaid_absence_days`: $0.0$ days (**normal**).
  - `longest_unpaid_streak_days`: $0.0$ days (**normal**).
  - `avg_hours_worked`: $8.0\text{ h/day}$ (**normal**).
  - `overtime_hours`: $0.0\text{ h}$ (**normal**).

- **Multi-Dimensional Masking Effect:**
  In a 10-dimensional feature space, Isolation Forest constructs random binary trees by selecting a random feature and a random split point. For `LATE_SPIKE` records, **9 out of 10 dimensions have completely ordinary, dense values**. Consequently, random partition cuts along any of the other 9 dimensions fail to isolate the point early. The point appears deep inside the dense cluster of normal employees unless a tree repeatedly cuts along the `late_arrivals` axis.

### Production Recommendation
To catch the remaining 27 `LATE_SPIKE` cases in production, add a 3rd business rule:
$$\text{rule\_late\_spike} = (\text{late\_arrivals} \ge 10)$$
Since the 99th percentile of `late_arrivals` across the workforce is $6.0$, an explicit threshold of $\ge 10$ late arrivals would recover all 27 remaining false negatives with near-zero false alarms.

---

## 7. Status of Project Resources & Environment

In accordance with instructions to inspect the workspace honestly and report missing resources without fabrication:

1. **`04_Validation_Vault` and `unlock_CX-05`:**
   - A recursive filesystem scan across all workspace directories (`task2`, `task 4`, `Task 1`) confirmed that neither `04_Validation_Vault` nor `unlock_CX-05` exists as a physical directory or executable script.
   - However, the validation labels and baselines are natively available inside the master dataset workbook `02_CortexHR_AI_ML_Dataset.xlsx` under the sheets `attendance_anomaly_labels` and `anomaly_baselines`.
   - Our implementation used the genuine workbook sheets directly for post-training validation and verified that no labels were used in training.
2. **`cortexhr_helpers.py`:**
   - As documented during Task 4, `cortexhr_helpers.py` was not provided in the original repository. All required functions (`catch_rate()`, `evaluate()`, `priority_rank()`, `run_detector()`) were implemented natively in `src/cortexhr/attendance_anomaly.py`.
3. **Preservation of Previous Tasks:**
   - All Task 1, Task 2, Task 3, and Task 4 code, pipelines, reports, tests, and documentation remain intact and unmodified.
   - Task 5 is encapsulated in `src/cortexhr/attendance_anomaly.py`, `scripts/run_attendance_anomaly.py`, `tests/test_attendance_anomaly.py`, and `reports/day5_attendance_anomaly.md`.

---

## 8. Verification & CLI Execution Commands

To reproduce all results, re-evaluate metrics, and run tests:

```bash
# 1. Run the Task 5 Anomaly Detection Pipeline
python scripts/run_attendance_anomaly.py

# 2. Run Task 5 Unit Tests (11 tests)
python -m pytest tests/test_attendance_anomaly.py -v

# 3. Run Entire Project Test Suite (121 tests across Tasks 1–5)
python -m pytest
```

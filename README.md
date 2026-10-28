# CortexHR — Phase 1: AI/ML Engineering & Analytics

**Candidate:** Dikshit Kumar  
**Track:** AI/ML  
**Repository:** [https://github.com/dikshit0012/CortexHR-](https://github.com/dikshit0012/CortexHR-)  
**Phase:** Phase 1 Review Submission  

---

## 📌 Project Overview

CortexHR is an end-to-end AI/ML-driven HR analytics system designed to predict employee attrition, detect data anomalies, prevent data leakage, and establish robust, reproducible machine learning pipelines.

Phase 1 encompasses Tasks 1 through 5, delivering data audits, feature pipelines, baseline modeling, evaluation under class imbalance, and attendance anomaly detection.

---

## 📂 Repository Structure

```text
Phase 1 All task/
├── Task 1/              # Data Audit & Exploration (Leakage, duplicates, data quality)
│   ├── cortexhr-ml/     # Task 1 codebase, EDA notebooks, reports
│   └── reports/         # Task 1 Data Audit Report (PDF & DOCX)
├── task 2/              # Feature Engineering & Reusable Pipeline
│   └── files/           # Clean scikit-learn ColumnTransformer, CV data joins
├── task 3/              # Honest Baseline Models & Metrics
│   └── files/           # 5-fold stratified CV, Decision Tree vs Logistic Regression
├── task 4/              # Model Evaluation & Diagnostic Analysis
│   └── files/           # Cost-matrix evaluation, ROC-AUC, Manager Note
├── task 5/              # Attendance Anomaly Detection & Phase 1 Final Review
│   ├── data/            # Raw datasets
│   ├── reports/         # Final Review Report (PDF & DOCX), anomaly metrics & flagged records
│   ├── scripts/         # Anomaly detector runner & report generator
│   ├── src/             # Anomaly detector module (Z-score + IQR + Isolation Forest)
│   └── tests/           # Unit and regression test suite
├── task 6/              # Phase 2 - Payroll Outlier Detector
│   ├── data/            # Local evaluation data is git-ignored
│   ├── reports/         # Metrics, masked flags, and comparison reports
│   ├── scripts/         # Pipeline runner and report verifier
│   ├── src/             # Payroll features, rules, model, and evaluation
│   └── tests/           # Task 6 automated tests
└── task 7/              # Phase 2 - Explainable flags and money-at-risk review
    ├── docs/            # Human-review checklist and feedback template
    ├── reports/         # Official-format output and review materials
    ├── scripts/         # Pipeline runner and report verifier
    ├── src/             # Reason codes, explanations, ranking, and risk estimates
    └── tests/           # Task 7 automated tests
```

---

## 🚀 Key Phase 1 Deliverables

| Task | Focus Area | Key Achievements & Deliverables |
|---|---|---|
| **Task 1** | Data Audit & Exploration | Detected 6 duplicate rows, 4 negative tenure errors, and 6 severe target leakage columns. Documented in `Task1_Data_Audit_Report.pdf`. |
| **Task 2** | Feature Pipeline | Built leak-free `ColumnTransformer` feature pipeline with time-series aggregates joined strictly inside CV folds. |
| **Task 3** | Honest Baselines | Evaluated Dummy classifier, balanced Logistic Regression (ROC-AUC ~0.712), and identified Decision Tree overfitting. |
| **Task 4** | Model Evaluation | Implemented business cost-matrix evaluation, threshold optimization, and manager briefing notes. |
| **Task 5** | Anomaly Detection & Review | Unsupervised multi-method anomaly detection (Isolation Forest, Z-score, IQR) with 100% test pass rate. Generated Phase 1 Comprehensive Review Report. |

---

## Phase 2 Deliverables

| Task | Focus Area | Key Deliverables |
|---|---|---|
| **Task 6** | Payroll anomaly detection | Payroll rules and Isolation Forest, per-error-type evaluation, and independently verified reports. |
| **Task 7** | Explainable anomaly review | Stable reason codes, plain-language explanations, money-at-risk estimates, and human-review materials. |

## 🧪 Running Tests & Scripts

To run the test suite for Task 5 anomaly detection:
```bash
pytest "task 5/tests"
```

To run the attendance anomaly detection pipeline:
```bash
python "task 5/scripts/run_task5.py"
```

To re-generate the Phase 1 Review Report:
```bash
python "task 5/scripts/generate_phase1_report.py"
```

To run the Phase 2 pipelines and their tests:
```bash
python -m pytest "task 6/tests"
python "task 6/scripts/run_task6.py"
python "task 6/scripts/verify_task6_reports.py"

python -m pytest "task 7/tests"
python "task 7/scripts/run_task7.py"
python "task 7/scripts/verify_task7_reports.py"
```

The Task 6 workbook and label mapping are local evaluation resources and are intentionally excluded from Git.

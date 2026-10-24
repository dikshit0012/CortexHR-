# Day 2 — Task 2: Reusable Feature Pipeline (evidence report)

Project: CortexHR employee attrition. Builds on the Task 1 audit ([`reports/Task1_Data_Audit_Report.pdf`](Task1_Data_Audit_Report.pdf), [`docs/decisions.md`](../docs/decisions.md), [`docs/open_items.md`](../docs/open_items.md)).

**Status:** pipeline implemented and tested. All 60 tests pass in the environment used (pandas 2.3.3, NumPy 2.3.5, scikit-learn 1.8.0, pytest 9.1.1, Python 3.12). The Docker setup has not been run. The GitHub repository could not be opened from this environment, so the work was done on a local copy and has not been pushed.

## 1. Code

| What | Path |
|---|---|
| `build_pipeline()`, feature selection, leakage guard, cleaner | [`src/cortexhr/features.py`](../src/cortexhr/features.py) |
| Monthly attendance/payroll aggregation and join | [`src/cortexhr/monthly.py`](../src/cortexhr/monthly.py) |
| Run the workflow (train/test split, fit on train only) | [`scripts/run_pipeline.py`](../scripts/run_pipeline.py) |
| Tests | [`tests/test_features.py`](../tests/test_features.py), [`tests/test_monthly.py`](../tests/test_monthly.py) |
| Machine-readable run summary | [`reports/tables/day2_pipeline_run.json`](tables/day2_pipeline_run.json) |

Once pushed, the same paths work on GitHub, e.g. `https://github.com/dikshit0012/context-hr/blob/<branch>/src/cortexhr/features.py` (branch name to be filled in; not verified).

```python
build_pipeline(include_unconfirmed: bool = False, use_monthly: bool = True, window_months: int = 3,
               numeric_features: Sequence[str] | None = None,
               categorical_features: Sequence[str] | None = None) -> sklearn.pipeline.Pipeline
```

## 2. What was inspected (actual files and columns)

* Main table `attrition_dataset`: 2,389 rows × 42 columns, target `left_within_12m`, identifiers `row_id` and `employee_id`, one date column used as a prediction date (`observation_date`) plus the exit dates `exit_date` and `resignation_date`.
* `attendance_monthly_features`: 7,339 rows; keys `employee_id` and `month`; columns used: `attendance_rate`, `unpaid_absence_days`. 0 duplicate employee-months.
* `payroll_monthly_features`: 7,355 rows; keys `employee_id` and `month`; column used: `variable_pay`. 15 exact duplicate employee-month rows (ids of the form `PI-D…`), removed.
* Both monthly tables cover only `EMP-` employees (528 ids, April 2025 to September 2026). All 383 master-snapshot employees in the attrition table have records; none of the 2,000 simulated `HIST-` rows do. About 5,000 rows in each table are dated after the 30 Sep 2025 snapshot.
* All six Task 1 leakage columns exist in the data and are confirmed to be post-outcome columns (see the Task 1 audit, section 8).

## 3. Feature-selection decisions

Model inputs: **33 columns** (confirmed-safe default).

* Categorical (6): `department_id`, `role_id`, `work_location`, `work_mode`, `employment_type`, `highest_education`
* Numeric (24): `job_level`, `tenure_months`, `total_experience_years`, `previous_employers_count`, `commute_distance_km`, `annual_fixed_pay`, `comp_ratio`, `raises_received`, `last_raise_pct`, `months_since_last_raise`, `promotions_received`, `months_since_last_promotion`, `internal_transfers`, `last_performance_rating`, `last_engagement_score`, `manager_changes_12m`, `training_hours_12m`, `attendance_rate_3m`, `late_arrivals_3m`, `avg_overtime_hours_per_month_3m`, `avg_hours_worked_3m`, `wfh_share_3m`, `unpaid_leave_days_6m`, `paid_leave_days_6m`
* Monthly, point-in-time (3): `attendance_rate_change_3m`, `avg_unpaid_absence_days_3m`, `avg_variable_pay_3m`

Excluded, and enforced by `ForbiddenColumnError`:

| Column(s) | Reason |
|---|---|
| `left_within_12m` | target |
| `exit_date`, `resignation_date`, `exit_reason`, `exit_interview_done`, `fnf_settlement_amount`, `voluntary_left_within_12m` | exist only after the person has left (leakage) |
| `row_id`, `employee_id` | identifiers |
| `source` | data-origin marker (real vs simulated) |
| `observation_date` | prediction date; used for the monthly windows and never as a feature |
| `dept_attrition_rate_prior_12m` | **unconfirmed** (section 8); available with `include_unconfirmed=True` |

Date columns: the only dates in the table are the observation date and the two exit dates. The exit dates are leaks, and the observation date is the reference point for the monthly windows, so no date column remains as a feature.

Row rules applied before splitting (`prepare_training_rows`): 6 duplicate rows dropped; 4 negative-tenure rows (EMP-0248, EMP-0280, EMP-0568, EMP-0596) excluded and **not corrected**, pending the Python team (open item O1). 2389 → 2379 rows.

## 4. Attendance and payroll aggregation

* **Join key:** `employee_id`, per attrition row (an `employee_id` + `observation_date` pair).
* **Window:** the 3 most recent completed months on or before `observation_date` (default `lag_months=0`). For 30 Sep 2025: 2025-07-01 to 2025-09-01. `window_bounds()` is unit tested for month-end, mid-month and lag cases.
* **Look-ahead guard:** after the join the code asserts that every month used ended on or before its row's observation date. Latest month used in this run: `2025-09-01`.
* **Duplicates:** exact duplicates are dropped ({'attendance_monthly_features': 0, 'payroll_monthly_features': 15}); conflicting duplicates raise an error.
* **Row count:** 2379 rows in, 2379 out (one-to-one join, checked).
* **Coverage in the window:** attendance months per row `{'0': 2000, '1': 3, '2': 10, '3': 366}` (key = months, value = rows); payroll `{'0': 2000, '1': 3, '2': 10, '3': 366}`. 2000 rows have no monthly data at all.
* **Missing months:** averages use the months that exist; a row with no months gets NaN (not zero) and a count of 0 months. Imputation is left to the pipeline (training median).
* **Target:** never read by `monthly.py`; a unit test checks it is returned unchanged.

## 5. The new feature

`attendance_rate_change_3m` = attendance rate of the latest month in the window − attendance rate of the earliest month in the window (columns `attendance_rate`, `month`).

* **Why useful:** a manager would act on attendance that is getting worse even if the average still looks acceptable.
* **No future information:** only completed months up to the observation date are used (tested with synthetic data containing future months with extreme values).
* **Missing:** NaN with fewer than two months (3 master-snapshot rows); filled with the training median inside the pipeline.
* **Evidence of value:** weak. Single-column AUC on the master-snapshot rows with a value: 0.528 (n = 376). The other two monthly inputs: `avg_unpaid_absence_days_3m` 0.562, `avg_variable_pay_3m` 0.480. These are quick checks on about 50 leavers, not model results. No sign of leakage, but also little signal.
* Also computed but **not** used as an input: `avg_attendance_rate_3m`, because it correlates r = 0.83 with the existing `attendance_rate_3m` column.

## 6. Pipeline architecture

```
raw rows ─► prepare_training_rows (drop duplicates, hold negative tenure)      [rows removed, outside sklearn]
        ─► add_monthly_features   (3 completed months before observation_date)
        ─► select_features        (allowed columns only; raises on target / leaks / ids)
        ─► Pipeline
             ├─ RawCleaner         city spelling, clip attendance & WFH to 0-1, experience >= tenure,
             │                     10x salary typo fix (role medians learned in fit), never_promoted/never_raised, log1p
             └─ ColumnTransformer
                  ├─ num      SimpleImputer(median, add_indicator=True) → StandardScaler
                  ├─ monthly  SimpleImputer(median) → StandardScaler
                  ├─ cat      SimpleImputer(constant "Unknown") → OneHotEncoder(handle_unknown="ignore")
                  └─ flags    passthrough (never_promoted, never_raised)
```

The pipeline itself refuses any input frame that still contains the target, a leak or an identifier.

## 7. Fitting on training data only

I fit the preprocessing pipeline on training data only so that information from the test set cannot influence imputation, scaling, or category encoding. This helps prevent data leakage and provides a more realistic evaluation of how the model performs on unseen employees. The fitted pipeline is then used to transform test data and future records without fitting it again.

In CortexHR the medians for blanks, scaling means and standard deviations, known categories and the per-role pay mid-points all come from the training split. `scripts/run_pipeline.py` does the split first (80/20, `stratify=y`, `random_state=42`), calls `pipe.fit(X_train)` once and only `transform` on the test set. Two tests check this: the scaler's statistics equal the training rows' statistics, and `transform` leaves the fitted values unchanged.

## 8. Run output

### 8.1 Pipeline run on the real workbook (`python scripts/run_pipeline.py`)

```
rows after Task 1 row rules : 2389 -> 2379 (6 duplicates, 4 negative-tenure rows left out)
monthly join                : 2379 rows in, 2379 out; 379 rows have monthly data, 2000 do not
payroll duplicate rows removed: 15
input columns               : 33   (target and forbidden columns excluded)
train / test                : (1903, 132) / (476, 132)   leaver rate 0.1466 / 0.1471
output columns              : 132   NaN: 0   inf: 0
saved fitted pipeline       : data/processed/feature_pipeline.joblib
```

Missing-value indicator columns created (numeric columns with blanks in the training split): `num__missingindicator_commute_distance_km`, `num__missingindicator_last_performance_rating`, `num__missingindicator_last_engagement_score`. Flag columns: `flags__never_promoted`, `flags__never_raised`. With `--include-unconfirmed` the run gives 34 input and 133 output columns, also with 0 NaN and 0 inf.

### 8.2 Three-row test output (synthetic rows in the real schema)

Command: `pytest tests/test_features.py::TestThreeRows -v -s`. The rows are synthetic (a normal record, a record full of blanks, and a messy record); this proves the code, not the real data.

```
THREE-ROW TEST (synthetic rows, real schema)
  raw columns in : 45   input columns after select_features: 33
  transformed    : shape (3, 43), dtype float64, NaN 0, inf 0
  output columns : 43 (numeric/scaled, missing-indicators, one-hot, flags)
  target y       : [0, 1, 0] (never passed to the pipeline)
```

### 8.3 Tests executed

Command: `pytest -v` (60 tests: 4 from Task 1, 56 for Day 2). Result: **60 passed in 6.24s**.

<details><summary>Full list</summary>

```
tests/test_audit.py::test_iqr_flags_the_extreme_value PASSED             [  1%]
tests/test_audit.py::test_z_mask_needs_a_big_sample_to_fire PASSED       [  3%]
tests/test_audit.py::test_iqr_overflags_zero_inflated_data_more_than_z PASSED [  5%]
tests/test_audit.py::test_location_aliases PASSED                        [  6%]
tests/test_features.py::TestThreeRows::test_three_row_end_to_end PASSED  [  8%]
tests/test_features.py::TestThreeRows::test_target_ids_and_leaks_are_not_inputs PASSED [ 10%]
tests/test_features.py::TestThreeRows::test_missing_values_are_handled PASSED [ 11%]
tests/test_features.py::TestThreeRows::test_scaling_and_one_hot_encoding PASSED [ 13%]
tests/test_features.py::TestThreeRows::test_unseen_categories_do_not_fail PASSED [ 15%]
tests/test_features.py::TestThreeRows::test_transform_new_records_after_fit PASSED [ 16%]
tests/test_features.py::TestThreeRows::test_raw_record_with_target_is_refused_by_pipeline PASSED [ 18%]
tests/test_features.py::TestRobustness::test_column_order_does_not_matter PASSED [ 20%]
tests/test_features.py::TestRobustness::test_extra_harmless_column_is_ignored PASSED [ 21%]
tests/test_features.py::TestRobustness::test_input_frame_is_not_modified PASSED [ 23%]
tests/test_features.py::TestRobustness::test_empty_input_gives_clear_error PASSED [ 25%]
tests/test_features.py::TestRobustness::test_completely_blank_monthly_columns_at_fit_give_clear_error PASSED [ 26%]
tests/test_features.py::TestRobustness::test_pipeline_can_be_cloned_and_saved PASSED [ 28%]
tests/test_features.py::TestRobustness::test_usable_inside_a_model_pipeline_and_target_stays_separate PASSED [ 30%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[left_within_12m] PASSED [ 31%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[exit_date] PASSED [ 33%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[resignation_date] PASSED [ 35%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[exit_reason] PASSED [ 36%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[exit_interview_done] PASSED [ 38%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[fnf_settlement_amount] PASSED [ 40%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[voluntary_left_within_12m] PASSED [ 41%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[row_id] PASSED [ 43%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[employee_id] PASSED [ 45%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[source] PASSED [ 46%]
tests/test_features.py::TestForbiddenColumns::test_selecting_forbidden_column_raises[observation_date] PASSED [ 48%]
tests/test_features.py::TestForbiddenColumns::test_custom_pipeline_lists_are_checked_too PASSED [ 50%]
tests/test_features.py::TestForbiddenColumns::test_missing_input_column_gives_helpful_error PASSED [ 51%]
tests/test_features.py::TestForbiddenColumns::test_unconfirmed_feature_is_off_by_default PASSED [ 53%]
tests/test_features.py::TestForbiddenColumns::test_feature_lists_match_data_dictionary PASSED [ 55%]
tests/test_features.py::TestForbiddenColumns::test_monthly_columns_follow_window PASSED [ 56%]
tests/test_features.py::TestFitOnTrainOnly::test_scaler_statistics_come_from_training_rows_only PASSED [ 58%]
tests/test_features.py::TestFitOnTrainOnly::test_transform_never_changes_fitted_state PASSED [ 60%]
tests/test_features.py::TestRawCleaner::test_task1_rules PASSED          [ 61%]
tests/test_features.py::TestRawCleaner::test_tenure_is_not_silently_corrected PASSED [ 63%]
tests/test_features.py::TestRawCleaner::test_non_numeric_text_in_numeric_column_is_reported PASSED [ 65%]
tests/test_features.py::TestRawCleaner::test_plain_arrays_are_rejected PASSED [ 66%]
tests/test_features.py::TestPrepareTrainingRows::test_duplicates_and_negative_tenure PASSED [ 68%]
tests/test_features.py::TestRealDataset::test_schema_and_row_counts PASSED [ 70%]
tests/test_features.py::TestRealDataset::test_three_real_rows_through_a_pipeline_fitted_on_train PASSED [ 71%]
tests/test_features.py::TestRealDataset::test_model_pipeline_runs_and_test_rows_are_only_transformed PASSED [ 73%]
tests/test_features.py::TestRealDataset::test_monthly_window_has_no_look_ahead PASSED [ 75%]
tests/test_monthly.py::TestWindow::test_last_day_of_month_includes_that_month PASSED [ 76%]
tests/test_monthly.py::TestWindow::test_mid_month_uses_only_completed_months PASSED [ 78%]
tests/test_monthly.py::TestWindow::test_lag_moves_window_back PASSED     [ 80%]
tests/test_monthly.py::TestWindow::test_bad_arguments PASSED             [ 81%]
tests/test_monthly.py::TestJoin::test_future_months_are_ignored PASSED   [ 83%]
tests/test_monthly.py::TestJoin::test_change_is_latest_minus_earliest PASSED [ 85%]
tests/test_monthly.py::TestJoin::test_one_month_gives_average_but_no_trend PASSED [ 86%]
tests/test_monthly.py::TestJoin::test_missing_employee_is_nan_not_zero PASSED [ 88%]
tests/test_monthly.py::TestJoin::test_no_row_multiplication_and_target_untouched PASSED [ 90%]
tests/test_monthly.py::TestJoin::test_partial_history_is_not_padded PASSED [ 91%]
tests/test_monthly.py::TestJoin::test_existing_column_clash_is_refused PASSED [ 93%]
tests/test_monthly.py::TestJoin::test_non_unique_index_is_refused PASSED [ 95%]
tests/test_monthly.py::TestDuplicates::test_exact_duplicate_months_are_removed PASSED [ 96%]
tests/test_monthly.py::TestDuplicates::test_conflicting_duplicates_raise PASSED [ 98%]
tests/test_monthly.py::TestDuplicates::test_duplicates_do_not_inflate_the_average PASSED [100%]
============================== 60 passed in 6.24s ==============================
```

</details>

Real-data checks (`TestRealDataset`, run, not skipped): schema 2,389 × 42; 2,379 rows after the join; 6 duplicates and 4 negative-tenure rows handled; three master-snapshot rows (two master-snapshot rows and one simulated row) pass through a pipeline fitted on the training split with finite output; no month after 2025-09-01 used; 15 payroll duplicates removed; 379 rows with and 2,000 rows without monthly data.

### 8.4 Do the tests actually catch mistakes?

I broke the code on purpose three times and re-ran the suite (a manual check done once, not an automated test; code restored afterwards, suite passed):

| Deliberate break | Result |
|---|---|
| Window no longer stops at the observation date | suite failed |
| Target no longer in the forbidden list | failed at import (a built-in consistency check) |
| `OneHotEncoder` set to raise on unseen categories | `test_unseen_categories_do_not_fail` failed |

(The 8 robustness tests added during the later audit were not mutation-checked.)

### 8.5 Task 1 regression check

`python scripts/run_audit.py` re-ran successfully after the changes; `findings.json` still reports 2,389 rows × 42 columns, 14.6% leavers and 24 decisions. The 4 Task 1 tests still pass.

## 9. Unresolved data questions and limitations

1. **`dept_attrition_rate_prior_12m` — cannot be confirmed safe (O2).** Real rows (383): constant within each department; mean gap to the prior-12-month exit rate from `monthly_headcount_history` 0.7 points vs 10.8 points to the following 12 months, which fits a backward-looking calculation. Simulated rows (2,000): the value differs row by row (D01 has 497 distinct values), does not follow the headcount sheet (mean gap about 6.5 points to either window, 400-row sample), so it could not be checked. Association with the target is about nil (AUC 0.53 simulated, 0.49 real), which makes direct leakage unlikely but does not settle the timing question. Decision: off by default; Task 3 should compare models with and without it. Question for the Python team: is it computed only from exits before each row's observation date, and how were the simulated values produced?
2. **Negative tenure (O1):** 4 rows excluded, join dates still needed.
3. **Is the September 2025 monthly data available on 30 Sep 2025?** Assumed yes (month counted once it has finished). `lag_months=1` gives a conservative alternative; its effect was not measured.
4. **Payroll duplicates:** 15 exact duplicate rows (`PI-D…` ids) removed by rule. Cause not documented.
5. **Monthly features exist for only 379 of 2,379 rows (16%).** For the other 84% they are the training median. Whether they add anything should be judged on the master-snapshot rows separately in Task 3.
6. **Simplifications versus Task 1 decisions:** global medians instead of department/city medians (D14/D15); "was missing" columns for engagement, rating and commute come from the imputer; no separate decision was made for `highest_education` spelling variants (27 raw values are one-hot encoded as they are).
7. **Split design:** random stratified split; simulated rows span 2021–2025, so a time-aware split may be more realistic.
8. **Not tested:** pandas 3, other scikit-learn versions, the Docker image, a scoring run on genuinely new employees.

## 10. Suggested commit message

`feat: build reusable employee attrition feature pipeline`

## 11. Later independent re-check

After this report was first written, the whole project was re-verified in [`reports/verification_audit.md`](verification_audit.md). That check found and fixed a missing `setuptools` dependency, a Windows encoding bug, a Docker/ignore-file issue and some documentation errors, added 8 tests, and recorded one extra limitation (the two row sources can be told apart with AUC 1.0). The numbers in sections 3 to 8 were not changed by it.

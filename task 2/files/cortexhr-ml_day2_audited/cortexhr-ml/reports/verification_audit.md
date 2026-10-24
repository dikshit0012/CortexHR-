# Verification audit: Day 1 (Data Audit) and Day 2 (Feature Pipeline)

Date of audit: 20 October 2026. Scope: the local copy of the CortexHR project in this environment.

**Read this first**

* **I could not open your GitHub repository** (`dikshit0012/context-hr`): cloning asks for a login, so it is private or not reachable from here. Everything below was checked on the local project, which matches the files you uploaded (README, requirements, Dockerfile, compose file, ignore files). I cannot say what is currently on your GitHub. **Nothing was committed or pushed.**
* **I wrote most of this code and report earlier in this conversation.** To avoid just agreeing with myself, I recomputed the Day 1 numbers with fresh code that does not use the project's helpers, and I tried to break the Day 2 pipeline with edge cases I had not tested before. This is still not the same as an independent human reviewer, so please run the commands in section 13 yourself.
* The workbook says everything in it is **fictional** (its README sheet). I treat it as such but still avoided committing it.

Status key: PASS = verified and working; FAIL = confirmed problem; PARTIALLY VERIFIED; NOT VERIFIED; NOT APPLICABLE.

---

## 1. Overall project status

| Area | Status | One-line summary |
|---|---|---|
| Day 1 data audit | **PASS** (after small fixes) | Every headline number reproduced independently. Some wording and a dictionary fact were wrong and are fixed. |
| Day 2 feature pipeline | **PASS** for how it works; **PARTIALLY VERIFIED** for leakage safety | 60 tests pass; no target leakage found; two things cannot be confirmed from the data (department rate, monthly-data timing). |
| Docker | **NOT VERIFIED** | Docker is not installed here. Files reviewed by reading only. |
| Windows | **PARTIALLY VERIFIED** | A real encoding bug was found and fixed (tested by emulation, not on Windows). |
| GitHub repo contents | **NOT VERIFIED** | No access. |

## 2. Repository structure review

| Check | Status | Evidence / problem |
|---|---|---|
| README, requirements, Dockerfile, compose, `.gitignore`, `.dockerignore` present | PASS | All present. |
| Folder layout matches README | PASS | `data/{raw,interim,processed}`, `notebooks/`, `src/cortexhr/`, `scripts/`, `tests/`, `reports/`, `docs/` all exist and are described. |
| Imports resolve, no broken imports | PASS | `python -m compileall` OK; ruff `F` and `B` rules: no findings after one fix (below). |
| Hardcoded paths | PASS | `grep` for `/home`, `C:\`, `/mnt` in `src/`, `scripts/`, `tests/`: none. Paths come from `src/cortexhr/paths.py` (relative to the repo root). |
| Dataset tracked by git | PASS | In a throw-away `git init` copy, `git ls-files` lists about 40 files and no `.xlsx`, `.joblib` or ydata `.html`. |
| Generated/large file tracked by git | **FAIL, fixed** | `reports/ydata_profile_attrition.html` (2.9 MB) contains a row-level "Sample" section and would have been tracked. Added to `.gitignore` and `.dockerignore`. |
| Dataset copied into the Docker image | **FAIL, fixed** | `Dockerfile` does `COPY . .` and `.dockerignore` did not exclude `data/`, so the workbook would be baked into image layers. Added `data/raw`, `data/processed`, `data/interim` to `.dockerignore` (not verified by building an image). |
| Missing dependency | **FAIL, fixed** | `ydata-profiling` 4.18 imports `pkg_resources` (from `setuptools`). In a clean Python 3.12 virtual environment the import failed. It only worked earlier because `setuptools` was already installed in my sandbox. Added `setuptools<81` to `requirements.txt` and re-tested. |
| `test` service sees edited tests | **FAIL (minor), fixed** | `docker-compose.yml` did not mount `./tests`. Added. |
| Notebook outputs contain employee rows/IDs | **PARTIALLY OK, your decision** | The notebook shows 21 distinct employee IDs and a few rows (e.g. `df.head(3)`). The data is fictional, so I left the outputs in as evidence. If your team treats the dataset as non-shareable, clear outputs before committing (`jupyter nbconvert --clear-output --inplace notebooks/01_data_audit.ipynb`). |
| Duplicate/unnecessary files | PASS | None found. `reports/day2_feature_pipeline.md` and this file are intentional. |
| "Part 4" required folder layout | NOT VERIFIED | You do not have the Part 4 instructions. |

## 3. Day 1 checklist

All numbers below were recomputed with new pandas code, not taken from `findings.json`.

| Requirement | Status | Evidence | Problem or next step |
|---|---|---|---|
| Loading and file validation | PASS | `run_audit.py` stops with a message if the workbook is missing; notebook reads `attrition_dataset` untouched. | none |
| Rows and columns | **PASS** | 2,389 rows × 42 columns (recomputed). | none |
| Names and dtypes; data dictionary | PASS | All 42 columns documented, none missing or extra (checked in code). | Fixed: dictionary said overtime is "75% zero"; actual is 81.3%. |
| Missing values | **PASS** | commute 55, education 15, promotion months 61.1%, raise % 8.5%, rating 182 (all with tenure < 12 months), engagement 22.1%. | Counts use the raw 2,389 rows while other figures use 2,383 de-duplicated rows. Now labelled in `decisions.md`. |
| Duplicates | **PASS** | 6 exact duplicates ignoring `row_id`; 6 repeated `employee_id`s. | none |
| Numerical summaries | PASS | Skew and IQR/z-score tables in `reports/tables/outlier_methods.csv`. | none |
| Outliers: salary, hours, tenure, two methods | **PASS** | IQR vs z-score: salary 93 vs 26, hours 48 vs 15, tenure 73 vs 45 (recomputed). | none |
| Genuine outlier vs data-entry error | **PASS** | Pay ÷ comp_ratio vs role median flags exactly 3 rows (EMP-0034, 0433, 0601; 11.1–11.4× the role median; next-highest is 1.18×). EMP-0313 (1.8 h/day, 20% attendance; next-lowest hours 6.65) kept and flagged. | Wording was too strong ("right where its job level belongs"). Corrected to "inside the range for its level": two of the three corrected salaries fall in the 5th–95th percentile of their level, EMP-0433 is slightly above it. This is an inference and still needs HR confirmation. |
| Target distribution and imbalance | **PASS** | 349 leavers; 14.61% of 2,389 or 14.65% of 2,383 (both round to 14.6%). | none |
| Leakage columns | **PASS** | `exit_date`, `exit_reason`, `exit_interview_done`, `fnf_settlement_amount` filled for 100% of leavers and 0% of stayers; `resignation_date` 92%; `voluntary_left_within_12m` = 1 for 321 of 349 leavers. | none |
| Suspicious/invalid values | **PASS** | 4 negative tenures (EMP-0248, 0280, 0568, 0596), 2 attendance above 1, 13 city-spelling variants (16 labels → 9), 213 WFH shares outside 0–1, 137 rows with experience below tenure (0 among master-snapshot rows). | none |
| Written findings and decisions | PASS | 24-row decision log, each with a reason. | none |
| Open questions | PASS | O1 (negative tenure), O2 (department rate), O3 (experience below tenure) with a response log. | none |
| The seven numbers you asked about | **PASS** | 2,389 × 42; 6; 4; 2; 55; 15; ~14.6%: all reproduced. | none |

**Smaller observations on Day 1.**

* The survey-response p-value (0.148) uses scipy's default continuity correction. Without it, p = 0.129. Same conclusion (not significant).
* Wording: the docs called the 383 `EMP-` employees "real". The workbook says everything is fictional. Changed to "master-snapshot" and added a note in the README.
* Treatments for missing values and duplicates are explained (keep, fill, flag, and why), not deleted blindly.

## 4. Day 2 checklist

| Requirement | Status | Evidence | Problem or next step |
|---|---|---|---|
| A. Target not in features | **PASS** | `select_features` and the pipeline raise `ForbiddenColumnError`; 16 tests in `TestForbiddenColumns` (11 parametrised over the target, the 6 leak columns and the 4 id/provenance columns, plus 5 more guard tests). | none |
| A. Identifiers not used | PASS | `employee_id`, `row_id` forbidden; `employee_id` is unique after preparation (2,379 rows). | none |
| A. Leaks and future info excluded | PASS | All six Task 1 leak columns forbidden. | see section 5 for what cannot be confirmed |
| A. Every selected feature exists | PASS | Selecting 33 columns from the real table works; a test compares the feature lists with the data dictionary. | none |
| A. Numeric/categorical classified correctly | PASS | 24 numeric + 3 monthly columns are numeric dtype; 6 categorical are `object` (checked on the real table). | none |
| B. Source tables and date fields identified | **PASS** | Attendance 7,339 rows, payroll 7,355 rows; keys `employee_id` + `month` (first-of-month dates). | none |
| B. Window documented and justified | PASS | 3 completed months before `observation_date`; `window_bounds()` tested (month-end, mid-month, lag). | The assumption that the latest month is known at the snapshot is not confirmable (section 6). |
| B. No future records | **PASS** | Latest month used = 2025-09-01; the join asserts that every month used ended on or before its observation date. 4,985 attendance and 4,996 payroll rows are dated after the snapshot and are ignored. | none |
| B. No row multiplication | **PASS** | 2,379 rows in, 2,379 out; one-to-one join checked; unit test with 18 months of data per employee. | none |
| B. Missing months | PASS | 366 rows have 3 months, 10 have 2, 3 have 1; 2,000 simulated rows have none and get NaN, not zero. | They are later filled with the training median. |
| B. Duplicates | PASS | 15 exact duplicate payroll rows removed; conflicting duplicates raise an error; attendance has none. | Cause of the 15 duplicates is unknown. |
| B. Reproducible | PASS | No randomness in `monthly.py`; repeated runs gave identical output. | none |
| C. Numeric imputation and scaling | PASS | Median imputer (+ missing indicators) then `StandardScaler`; tested. | none |
| C. Categorical imputation and one-hot | PASS | Constant "Unknown" + `OneHotEncoder(handle_unknown="ignore")`. | none |
| C. Unseen categories | **PASS** | Unseen role/department gives all-zero one-hot columns; mutation check (switching to `handle_unknown="error"`) made the test fail, as it should. | none |
| C. Train/inference consistency | PASS | Same output after `joblib` save/load; column order and extra harmless columns do not change the output. | none |
| C. Empty/missing/unexpected columns | **PASS after fix** | Missing required column: clear `KeyError`. Extra identifier column: clear `ForbiddenColumnError`. **Empty input and fully blank monthly columns gave unhelpful sklearn errors; fixed with clear messages and tests.** | none |
| C. scikit-learn compatibility | **PASS** | 60 tests pass on scikit-learn 1.4.0 (minimum allowed), 1.8.0 and 1.9.1; pandas 2.2.0 and 2.3.3; NumPy 1.26.4 and 2.3.5. | pandas 3 not tried. |
| D. `build_pipeline()` importable, real schema | PASS | `from cortexhr.features import build_pipeline`. | Lives in `src/cortexhr/features.py`, not `src/features.py`, matching the existing package layout. |
| D. Preprocessing-only | PASS | Returns `Pipeline[RawCleaner, ColumnTransformer]`, no classifier. | none |
| D. Target not needed at inference | PASS | `X` never contains the target. | none |
| D. Does not fit on test/inference data | PASS | Test: scaler mean equals the training rows' mean; `transform` leaves fitted values unchanged. | none |
| D. Works with a model | PASS | `Pipeline([features, LogisticRegression])` fits and predicts on synthetic and real data. | none |
| E. Fit on train only | PASS | `run_pipeline.py` splits first (stratified), then fits on train only. | none |
| E. Global preprocessing before the split | PASS | Only de-duplication, the negative-tenure exclusion and the monthly join happen before the split; none use the target or cross-row statistics. | none |

## 5. Data leakage review

**Target leakage: none found.**

* The target is only ever read as `y`. No column in the selected inputs is a function of it.
* Confirmed leak columns are excluded and enforced by code and tests.
* Single-column AUC against the target for the monthly features is 0.48–0.56, for `dept_attrition_rate_prior_12m` 0.53 (simulated rows) and 0.49 (master-snapshot rows). Nothing resembles a leak (a leak would be near 1.0).

**Future-data leakage: none found, with two caveats.**

1. **`dept_attrition_rate_prior_12m` cannot be confirmed safe (open item O2).** For the 383 master-snapshot rows it is constant per department and 0.7 points from the prior-12-month exit rate in the headcount sheet (10.8 points from the following 12 months). For the 2,000 simulated rows it varies row by row (e.g. 497 distinct values in one department) and does not follow the headcount sheet (about 6.5 points off either window in a 400-row sample), so it could not be checked. It is off by default.
2. **Monthly data timing.** The window assumes the September 2025 figures were known on 30 Sep 2025. `lag_months=1` is the conservative alternative; its effect was not measured.

**Not leakage, but a risk to evaluation: the two row sources are easy to tell apart.** A random forest separates master-snapshot rows from simulated rows with AUC 1.0 on the pipeline output (0.998 without the monthly features), mainly via `attendance_rate_3m`, `months_since_last_raise` and the monthly features. Their leaver rates differ (13.2% vs 15.0%) but not significantly (p = 0.42). Consequences: a model may partly learn "which source is this", and the monthly features are constant for all simulated rows, which makes them a source marker. In a quick logistic-regression check (not a Task 3 result) the cross-validated AUC was 0.68 on master-snapshot rows and 0.72 on simulated rows. **Report metrics per source in Task 3.**

*Correction of an earlier claim:* an earlier version of the README said the monthly features have no "was missing" flag because a flag would reveal the data source. That reasoning was wrong, since the constant median fill reveals it too. The README, the code comment and the Day 2 report now say so.

## 6. Attendance and payroll aggregation review

| Item | Status | Detail |
|---|---|---|
| Actual columns used | PASS | `attendance_rate`, `unpaid_absence_days` (attendance); `variable_pay` (payroll); `employee_id`, `month` keys. |
| Window | PASS | 3 most recent completed months on or before `observation_date`. |
| Coverage | PARTIALLY VERIFIED | Only 379 of 2,379 rows (16%) have any monthly data; they are all master-snapshot rows. The 2,000 simulated rows have none, so the monthly features can only help on about 379 rows (about 50 leavers). |
| `avg_attendance_rate_3m` | NOTE | Computed but not used as an input: correlation with the existing `attendance_rate_3m` is 0.83, and why they differ is undocumented. |
| Strength of the new features | NOTE | Weak (single-column AUC 0.53 for `attendance_rate_change_3m`). No leakage sign, but also little signal. |

## 7. Test results

| Test | Method | Result | Remaining issue |
|---|---|---|---|
| Feature module imports | `pytest` collection; `from cortexhr.features import build_pipeline` | **PASS** | none |
| `build_pipeline()` runs | `scripts/run_pipeline.py` | **PASS** (132 output columns, 0 NaN, 0 inf) | none |
| Three raw records transform (synthetic) | `TestThreeRows` (7 tests) | **PASS**, output shape (3, 43) | Synthetic rows prove the mechanics only. |
| Three real rows transform | `TestRealDataset` | **PASS** (two master-snapshot rows and one simulated row through a pipeline fitted on the training split) | none |
| Row count preserved | same | PASS | none |
| Numeric, finite output | same | PASS | none |
| Missing values handled | `test_missing_values_are_handled` | PASS | none |
| Unseen category | `test_unseen_categories_do_not_fail` | PASS (and fails if the encoder is changed to raise) | none |
| Target not in features | `test_target_ids_and_leaks_are_not_inputs` | PASS | none |
| Leak columns rejected | `TestForbiddenColumns` (16) | PASS | none |
| No hardcoded path | `grep` | PASS | none |
| Feature names consistent | `test_transform_new_records_after_fit` and name checks | PASS | none |
| Dependencies sufficient | fresh venv from `requirements.txt` | **PASS after fix** (needed `setuptools<81`) | none |
| Whole suite | `pytest -q` | **60 passed**: system Python (sklearn 1.8.0), clean venv (sklearn 1.9.1), minimum-version venv (sklearn 1.4.0) | none |
| Task 1 not broken | `run_audit.py`; findings compared with the previous run | PASS (no finding changed; ydata profile now generated in a clean venv) | none |
| Lint | `ruff` (E, F, W, B) | `F` and `B`: clean after removing an unused variable. 11 `E501` long-line warnings remain (style, mostly dictionary text). | cosmetic |
| Windows-like encoding | `run_audit.py`, `run_pipeline.py`, tests with a non-UTF-8 default | **FAIL before fix, PASS after** (UnicodeEncodeError when writing `docs/*.md`; fixed with `encoding="utf-8"`) | Not run on real Windows. |
| Mutation check (3 deliberate breaks) | manual | PASS: each break was caught | The 8 later robustness tests were not mutation-checked. |
| Docker commands | not runnable (no Docker) | **NOT RUN** | run locally |

Test counts by file: `test_audit.py` 4; `test_features.py` 41 (ThreeRows 7, ForbiddenColumns 16, FitOnTrainOnly 2, RawCleaner 4, PrepareTrainingRows 1, Robustness 7, RealDataset 4); `test_monthly.py` 15 (Window 4, Join 8, Duplicates 3). Total 60. The 8 tests added during this audit are the 7 in `TestRobustness` plus 1 real-data model-pipeline test. The exact per-test list is in `reports/day2_feature_pipeline.md`.

## 8. Documentation review

| Check | Status | Note |
|---|---|---|
| Setup instructions | PARTIALLY VERIFIED | Local venv path verified by a clean install; Windows PowerShell commands read but not run; Docker not run. |
| Dataset path | PASS | `data/raw/02_CortexHR_AI_ML_Dataset.xlsx`, git-ignored. |
| Commands accurate | PASS | `pytest -q` works without `PYTHONPATH` (`pytest.ini`); `python scripts/run_pipeline.py` and `run_audit.py` verified. |
| Three-row test described honestly | PASS | Labelled synthetic; a separate real-data class exists. |
| Leakage prevention documented | PASS | README and Day 2 report. |
| Unsupported claims | **FOUND, fixed** | Overtime "75% zero"; "real employees"; "right where its job level belongs"; the monthly-flag reasoning. |
| Unresolved questions documented | PASS | README, `docs/open_items.md`, Day 2 report section 9. |
| Report numbers vs code | PASS | Day 1 and Day 2 numbers match the recomputation. |

## 9. Docker and Windows compatibility

* **Docker: NOT VERIFIED.** `docker` is not installed here. I reviewed `Dockerfile` and `docker-compose.yml` by reading and parsed the compose file as YAML (services: `jupyter`, `audit`, `pipeline`, `test`). The documented commands match those services. Risks I could not test: (a) `USER appuser` plus bind mounts can fail to write `reports/` on some Linux hosts (user-id mismatch; usually fine on Docker Desktop for Windows); (b) the image build itself; (c) Jupyter runs with no token (local use only, as the Dockerfile says).
* **Windows: PARTIALLY VERIFIED.** The PowerShell commands in the README look right (`.\.venv\Scripts\Activate.ps1`, `$env:PYTHONPATH="src"`); a first-time run may need `Set-ExecutionPolicy -Scope Process RemoteSigned` to allow the activation script. The encoding bug that would have broken the audit on Windows is fixed and was tested by emulation only.

## 10. Problems discovered

| # | Severity | Problem | Status |
|---|---|---|---|
| 1 | High | `ydata-profiling` import fails in a clean Python 3.12 environment (missing `setuptools`) | fixed, verified |
| 2 | High | Audit notebook crashes under a non-UTF-8 default encoding (Windows) | fixed, verified by emulation |
| 3 | High | Dataset would be copied into the Docker image | fixed (not build-tested) |
| 4 | Medium | 2.9 MB HTML profile with sample rows would be committed | fixed (git-ignored) |
| 5 | Medium | Two row sources are trivially separable (AUC 1.0); metrics should be reported per source | documented, needs action in Task 3 |
| 6 | Medium | Day 2 doc gave a wrong reason for omitting the monthly "was missing" flag | corrected |
| 7 | Medium | Empty input and fully blank monthly columns gave opaque errors | fixed, tested |
| 8 | Low | Dictionary said overtime is 75% zero (it is 81%) | fixed |
| 9 | Low | "real employees" wording; overstated salary-typo wording; undocumented count denominators | fixed |
| 10 | Low | Unused variable in `monthly.py`; `tests/` not mounted in compose | fixed |
| 11 | Info | 11 long-line style warnings | left as is |

## 11. Fixes applied

Files changed in this audit: `requirements.txt`, `.gitignore`, `.dockerignore`, `docker-compose.yml`, `README.md`, `src/cortexhr/features.py`, `src/cortexhr/monthly.py`, `src/cortexhr/data_dictionary.py`, `scripts/run_pipeline.py`, `tests/test_features.py` (8 tests added), regenerated `notebooks/01_data_audit.ipynb`, `docs/*.md`, `reports/Task1_Data_Audit_Report.docx/.pdf`, `reports/day2_feature_pipeline.md`, and this file. No existing result was changed to look better: all Task 1 findings are identical before and after.

## 12. Remaining issues

1. Python team answers needed: O1 (join dates for 4 employees), O2 (department rate timing, and how the simulated values were produced), whether September 2025 monthly data was available on 30 Sep 2025, and why 15 payroll rows are duplicated.
2. Docker build and run, and a real Windows run, not verified.
3. Whether the notebook outputs and the HTML profile may be shared in your team's repository.
4. Source mix: plan per-source evaluation in Task 3.
5. pandas 3 untested; the exact GitHub state unknown.

## 13. Exact commands to run locally

Windows PowerShell:

```powershell
cd path\to\context-hr
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
Copy-Item "C:\path\to\02_CortexHR_AI_ML_Dataset.xlsx" data\raw\
pytest -q                              # expect: 60 passed
pytest tests/test_features.py::TestThreeRows -v -s
python scripts/run_pipeline.py
python scripts/run_audit.py            # re-runs the Task 1 notebook (about a minute)
```

Docker:

```bash
docker compose build
docker compose run --rm test
docker compose run --rm pipeline
docker compose run --rm audit
```

Before committing: `git status` must not list the `.xlsx`, `data/processed/*.joblib` or `reports/ydata_profile_attrition.html`.

## 14. Final submission checklist

- [ ] Run section 13 locally; confirm 60 passed and the pipeline output matches `reports/day2_feature_pipeline.md`
- [ ] Docker commands run (or say they were not run)
- [ ] `git status` shows no dataset, joblib or HTML profile
- [ ] Decide whether to clear notebook outputs
- [ ] Compare the folder layout with Part 4 when you have it
- [ ] Keep O1/O2 open and unanswered in the report unless the Python team replies
- [ ] You commit and push yourself (nothing was pushed)

---

## Direct answers

1. **Is my Day 1 work correct?** Yes. All the headline numbers reproduce independently. A few wordings and one dictionary figure were wrong; they are fixed.
2. **Is my Day 2 work correct?** Yes for the implementation and tests. Leakage safety is only partly verifiable: `dept_attrition_rate_prior_12m` and the timing of the monthly data cannot be confirmed from the data.
3. **Does the feature pipeline actually work?** Yes: 132 finite numeric columns on 2,379 rows, 60 tests passing on three scikit-learn versions, usable inside a classifier pipeline.
4. **Any target or future-data leakage?** No target leakage found. No future-data leakage found in the monthly join. The department rate is unconfirmed (off by default). The two row sources are easy to tell apart, which is an evaluation risk, not a leak.
5. **Are the tests sufficient?** For Task 2, mostly. They cover the checklist, the guards, the join and the edge cases I tried. Missing: Docker and Windows runs, mutation checks for the 8 newest tests, and any test of model quality (that is Task 3).
6. **Anything essential missing before submission?** Nothing in the code. You still need to run it yourself, check Docker, push to GitHub, and decide on the notebook outputs.
7. **Three things to fix first:** (1) run the commands in section 13 on your machine, including Docker; (2) make sure the HTML profile and dataset are not committed, and decide about notebook outputs; (3) get the Python team's answers on the department rate and the monthly-data timing, and plan per-source metrics for Task 3.

# ML Pipeline — Extending the Machine Learning Training Pipeline

Use this prompt when adding new algorithms, metrics, preprocessing steps, or modifying the training workflow.

## Context

@workspace The ML training pipeline architecture:

1. **Route**: POST `/models/<album_id>` → `webapp/routes/models.py`
2. **Service**: `webapp/service/machine_learning.py` → `train_model()`
   - Fetches features from DB, encodes labels, splits train/test
   - Instantiates `Classification` or `Survival` subclass
   - Calls `create_model()` which dispatches `quantimage2tasks.train` to the `training` queue
3. **Worker**: `workers/tasks.py` → `train_model()`
   - Runs `GridSearchCV` with pipeline, param grid, and scoring
   - Computes bootstrap confidence intervals (100 iterations for test, quartile-based for CV)
   - Computes `permutation_importance` (10 repeats)
   - Saves model via `joblib.dump()` to `/quantimage2-data/models/`
   - Creates `Model` DB record with metrics, predictions, feature importances

## Architecture

**Base class**: `webapp/modeling/modeling.py` → `Modeling` (abstract)

**Subclasses**:
- `webapp/modeling/classification.py` → `Classification`
  - Algorithms: `logistic_regression`, `svm`, `random_forest`
  - CV: `RepeatedStratifiedKFold` (5 folds, 1 repeat)
  - Scoring: AUC, accuracy, precision, sensitivity, specificity
  - Refit: AUC
- `webapp/modeling/survival.py` → `Survival`
  - Algorithms: `cox`, `cox_elastic`, `ipc`
  - CV: `SurvivalRepeatedStratifiedKFold` (custom, stratifies on Event bool)
  - Scoring: C-index via `concordance_index_censored`
  - Refit: C-index

**Pipeline pattern**:
```python
Pipeline([
    ("preprocessor", StandardScaler()),  # or MinMaxScaler, or None
    ("classifier", estimator),           # "analyzer" for survival
])
```

**Normalization** (`webapp/modeling/utils.py`): `StandardScaler` or `MinMaxScaler`

**Estimator step names** (`const.py`): `ESTIMATOR_STEP.CLASSIFICATION = "classifier"`, `ESTIMATOR_STEP.SURVIVAL = "analyzer"`

## Key Files

- `webapp/modeling/modeling.py` — Abstract `Modeling` base class with `__init__()`, `create_model()`
- `webapp/modeling/classification.py` — Classification algorithms, param grids, scoring
- `webapp/modeling/survival.py` — Survival algorithms, param grids, custom CV/scoring
- `webapp/modeling/utils.py` — `get_normalization_options()` (scaler grid)
- `webapp/service/machine_learning.py` — `train_model()` orchestration
- `workers/tasks.py` — `train_model()` Celery task (GridSearchCV, bootstrap, importance)
- `shared/quantimage2_backend_common/modeling_utils.py` — `SurvivalRepeatedStratifiedKFold`, `c_index_score`
- `shared/quantimage2_backend_common/const.py` — `MODEL_TYPES`, `ESTIMATOR_STEP`, `CV_SPLITS`, `CV_REPEATS`
- `shared/quantimage2_backend_common/models.py` — `Model` ORM class

## Rules for Adding a New Algorithm

1. Add to the appropriate subclass (`Classification` or `Survival`).
2. Define the estimator, parameter grid, and pipeline step name.
3. Follow the existing pattern: `get_estimator_grid()` returns `{step_name: [estimator], step_name__param: [values]}`.
4. Use `GridSearchCV` — never manually loop over hyperparameters.
5. Ensure the estimator is compatible with `sklearn.pipeline.Pipeline`.
6. For survival: use `sksurv` estimators and the custom `c_index_score` scorer.
7. For classification: wrap SVMs in `CalibratedClassifierCV` for probability support.

## Rules for Adding a New Metric

1. Define the scorer in the subclass's `get_scoring()` method.
2. Use `sklearn.metrics.make_scorer()` with appropriate parameters.
3. Add the metric key to the subclass's scoring dict.
4. Update `workers/tasks.py` to extract and store the new metric from `GridSearchCV` results.
5. Update the `Model` ORM class if new DB columns are needed.
6. Compute bootstrap confidence intervals for the new metric (follow existing pattern).

## Rules for New Preprocessing Steps

1. Add to `webapp/modeling/utils.py` → `get_normalization_options()`.
2. Follow the pattern: `{"preprocessor": [ScalerClass()]}` in param grid.
3. New scalers must be sklearn-compatible transformers.

## Bootstrap & Confidence Intervals

- Training metrics: quartile-based CI from CV fold scores.
- Test metrics: 100 bootstrap iterations of resampled test set → quartile-based CI.
- Store as `{mean, inf_value, sup_value}` in the `Model` DB record.

## Model Comparison

- `mlxtend.evaluate.permutation_test` on bootstrap AUC/C-index scores.
- Endpoint: POST `/models/compare` and POST `/models/compare-data`.

## Clinical Feature Integration

- Clinical features are merged with radiomic features before training.
- Types: `Number`, `Categorical`.
- Encodings: `None`, `One-Hot Encoding`, `Normalization`, `Ordered Categories`.
- Missing values: `Drop`, `Mode`, `Median`, `Mean`, `None`.
- The `Modeling.__init__()` method handles merging and encoding.

# ML Modeling Pipeline

## ML Modeling Pipeline

### Architecture

Abstract base class `Modeling` (`webapp/modeling/modeling.py`) with subclasses:
- `Classification` (`classification.py`) — binary classification
- `Survival` (`survival.py`) — survival analysis

### Classification Algorithms

| Algorithm | Key | Hyperparameter Grid |
|---|---|---|
| Logistic Regression | `logistic_regression` | solver: [lbfgs, saga], penalty: [l1, l2, elasticnet], C: [0.001–100] |
| SVM | `svm` | C: [0.01–100], gamma: [scale, auto, 1, 0.1, 0.01, 0.001], kernel: [linear, rbf, poly, sigmoid] |
| Random Forest | `random_forest` | max_depth: [10, 100, None], n_estimators: [10, 100, 1000] |

**CV**: `RepeatedStratifiedKFold` (5 folds, 1 repeat)
**Scoring**: AUC (roc_auc), accuracy, precision, sensitivity (recall), specificity
**Refit metric**: AUC

### Survival Algorithms

| Algorithm | Key | Library |
|---|---|---|
| Cox PH | `cox` | `sksurv.linear_model.CoxPHSurvivalAnalysis` |
| CoxNet | `cox_elastic` | `sksurv.linear_model.CoxnetSurvivalAnalysis` |
| IPC Ridge | `ipc` | `sksurv.linear_model.IPCRidge` |

**CV**: `SurvivalRepeatedStratifiedKFold` (custom, stratifies on Event bool only)
**Scoring**: C-index via `concordance_index_censored`
**Refit metric**: C-index

### Training Flow

1. **Route**: POST `/models/<album_id>` → `routes/models.py`
2. **Service**: `service/machine_learning.py` → `train_model()`
   - Fetches features, encodes labels, splits train/test
   - Creates `Modeling` subclass instance (Classification or Survival)
   - Calls `create_model()` which dispatches `quantimage2tasks.train` to `training` queue
3. **Worker**: `workers/tasks.py` → `train_model()`
   - Runs `GridSearchCV` with pipeline, param grid, scoring
   - Computes bootstrap confidence intervals (100 iterations for test metrics)
   - Computes `permutation_importance` (10 repeats)
   - Saves model via `joblib.dump()` to `/quantimage2-data/models/`
   - Creates `Model` DB record with all metrics and predictions

### Pipeline Structure

```python
Pipeline([
    ("preprocessor", StandardScaler()),  # or MinMaxScaler, or None
    ("classifier", LogisticRegression()), # or "analyzer" for survival
])
```

The estimator step name is `"classifier"` for classification, `"analyzer"` for survival (defined in `const.py` as `ESTIMATOR_STEP`).

### Normalization Options

| Method | Scaler |
|---|---|
| `standardization` | `StandardScaler` |
| `minmax` | `MinMaxScaler` |

### Clinical Feature Integration

Clinical features are merged with radiomic features before training (`get_clinical_features` in `service/machine_learning.py`):
- Types: `Number`, `Categorical`
- Encodings: `None`, `One-Hot Encoding`, `Normalization`, `Ordered Categories`
- Missing values: `Drop`, `Mode`, `Median`, `Mean`, `None`
- **Multi-CSV:** each uploaded file (`ClinicalFeatureFile`) owns its definitions, so the same column name can exist in two files. Clinical feature columns in the merged matrix are namespaced `{clinical_feature_file_id}::{name}` (`CLINICAL_FEATURE_ID_SEPARATOR`) to keep them distinct.
- A `FeatureCollection`'s `feature_ids` may store clinical features in either form: `{file_id}::{name}` (current) or a bare `{name}` (legacy, pre multi-CSV). `resolve_collection_clinical_definitions()` accepts both and de-duplicates by definition id; a legacy bare name resolves to a **single** definition — the one in the lowest `clinical_feature_file_id` (the migration-backfilled "Legacy" file) — so an old collection never silently pulls the same feature from several files.

### Model Comparison

`mlxtend.evaluate.permutation_test` on bootstrap AUC/C-index scores between two models.

### Rules for ML Code

- Use the existing `Pipeline` pattern — preprocessor + estimator.
- New algorithms must be added as options in `Classification` or `Survival` subclass.
- Always use `GridSearchCV` for hyperparameter tuning — do not manually loop.
- Compute bootstrap confidence intervals for all metrics.
- Save models via `joblib.dump()` — never use `pickle` directly.
- Feature importance must use `permutation_importance` for consistency.
- Survival label encoding must use `sksurv.util.Surv.from_dataframe()`.

---


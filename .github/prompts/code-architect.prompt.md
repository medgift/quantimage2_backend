# Code Architect — Architecture Consistency Review

Use this prompt to review code for compliance with the project's established patterns, conventions, and architecture.

## Context

@workspace Review the code for consistency with the QuantImage v2 backend architecture. Check the following areas:

## Architecture Checklist

### 1. Route/Blueprint Pattern

- [ ] Blueprint has `@bp.before_request` → `validate_decorate(request)`
- [ ] Routes use `g.user` for user scoping and `g.token` for Kheops forwarding
- [ ] Responses use `jsonify()` with appropriate HTTP status codes
- [ ] Error handling uses `InvalidUsage` (400) or `ComputationError` (500)
- [ ] Blueprint registered in `webapp/app.py` → `setup_app()`

### 2. ORM Model Pattern

- [ ] Models inherit from `BaseModel, db.Model`
- [ ] `__init__()` sets all required fields
- [ ] Has `to_dict()` method for JSON serialization
- [ ] Uses `BaseModel` CRUD methods (`save_to_db`, `find_by_id`, `get_or_create`, etc.)
- [ ] Foreign keys defined with `db.ForeignKey()`
- [ ] Queries use `db.session.query()` or class methods, not raw SQL (unless performance-critical)

### 3. Celery Task Pattern

- [ ] `bind=True` for access to `self`
- [ ] Explicit `queue=` parameter (`"extraction"` or `"training"`)
- [ ] `time_limit` and `soft_time_limit` set for extraction tasks
- [ ] `serializer="pickle"` only for training tasks
- [ ] Progress reported via Socket.IO (not just Celery state)
- [ ] `SoftTimeLimitExceeded` handled gracefully
- [ ] Temp files cleaned up in `finally` blocks

### 4. Feature ID Convention

- [ ] Uses `FEATURE_ID_SEPARATOR` (U+2011 non-breaking hyphen) — never regular `-`
- [ ] Feature names start with known prefix (PyRadiomics, Riesz, or ZRAD)
- [ ] Parsed via `featureIDMatcher` regex from `const.py`

### 5. ML Pipeline Pattern

- [ ] Uses `sklearn.pipeline.Pipeline` with `("preprocessor", scaler)` + `("classifier"|"analyzer", estimator)`
- [ ] Hyperparameter tuning via `GridSearchCV` — no manual loops
- [ ] Bootstrap confidence intervals computed for all metrics
- [ ] Feature importance via `permutation_importance`
- [ ] Models saved via `joblib.dump()` — never raw `pickle`
- [ ] Survival labels encoded via `sksurv.util.Surv.from_dataframe()`

### 6. Kheops Integration

- [ ] Uses `get_token_header(token)` for all Kheops requests
- [ ] URLs constructed from `KheopsEndpoints` constants — no hardcoded URLs
- [ ] Response status checked before `.json()`
- [ ] DICOM fields accessed via `DicomFields` hex tag constants

### 7. Security & Medical Data

- [ ] No patient names, JWT tokens, or passwords logged
- [ ] All DB queries scoped by `g.user` (unless admin-only)
- [ ] Temp DICOM files deleted after use
- [ ] Docker secrets for credentials — not env vars
- [ ] User-supplied paths validated with `pathvalidate`

### 8. Code Style

- [ ] Type hints on new function signatures
- [ ] `f-strings` for formatting (no `%` or `.format()`)
- [ ] `logging` module (not `print()`) in new code
- [ ] Imports grouped: stdlib → third-party → local
- [ ] `pathlib.Path` for filesystem ops in new code
- [ ] Black-formatted

### 9. Docker & Networking

- [ ] Service names used for inter-container communication (not IPs)
- [ ] New services added to base `docker-compose.yml` + all overlays
- [ ] Shared volume `backend-data` mounted where needed
- [ ] Health checks for dependency services

## Key Reference Files

- `webapp/routes/features.py` — Canonical route pattern
- `webapp/modeling/classification.py` — Canonical ML pipeline pattern
- `workers/tasks.py` — Canonical Celery task pattern
- `shared/quantimage2_backend_common/models.py` — Canonical ORM model pattern
- `shared/quantimage2_backend_common/const.py` — All constants and enums

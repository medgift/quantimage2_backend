# QuantImage v2 Backend – Copilot Instructions

## Project Overview

QuantImage v2 is a **radiomics research platform** for medical imaging. This repository is the **backend**, written in Python, running across multiple Docker containers. It provides:

- A **Flask REST API** + **Socket.IO** server for the frontend SPA
- **Celery workers** for asynchronous feature extraction (PyRadiomics/Okapy) and ML model training
- **MySQL** database for features, labels, models, and metadata
- **Redis** for Celery broker/result backend and Socket.IO message queue
- Integration with **Kheops** (PACS/DICOMweb) for medical image storage and **Keycloak** for OIDC authentication

### Data Flow

```
Frontend (React SPA)
  → Backend API (Flask, port 5000)
    → Kheops (DICOMweb API): download DICOM studies
    → Celery Workers:
        extraction queue → PyRadiomics/Okapy → feature values stored in MySQL
        training queue   → scikit-learn / scikit-survival → model files saved to disk
    → Socket.IO: real-time progress updates to frontend
```

---

## Repository Structure

```
quantimage2_backend/
├── shared/                              # Shared library (quantimage2_backend_common)
│   ├── setup.py
│   └── quantimage2_backend_common/
│       ├── const.py                     # Enums, constants, feature ID regex
│       ├── feature_storage.py           # Store/retrieve features from MySQL
│       ├── flask_init.py                # Flask app factory, DB connection
│       ├── kheops_utils.py              # Kheops API client, DICOM field mappings
│       ├── modeling_utils.py            # Survival CV splitter, c-index scorer
│       ├── models.py                    # SQLAlchemy ORM models (1670 lines)
│       └── utils.py                     # Error classes, formatters, Socket.IO helpers
├── webapp/                              # Flask web application
│   ├── app.py                           # App entry point, blueprint registration
│   ├── config.py                        # Keycloak client, Flask config
│   ├── populate.py                      # Seed default feature presets
│   ├── routes/                          # Flask Blueprints (REST endpoints)
│   │   ├── utils.py                     # Auth decorators (decode_token, validate_decorate)
│   │   ├── features.py                  # /extractions/*, /extract/*
│   │   ├── models.py                    # /models/*
│   │   ├── albums.py                    # /albums/*
│   │   ├── tasks.py                     # /tasks/*
│   │   ├── labels.py                    # /labels/*
│   │   ├── charts.py                    # /charts/*
│   │   ├── feature_presets.py           # /feature-presets/*
│   │   ├── feature_collections.py       # /feature-collections/*
│   │   ├── clinical_features.py         # /clinical-features/*
│   │   └── navigation_history.py        # Navigation tracking
│   ├── modeling/                         # ML pipeline
│   │   ├── modeling.py                  # Abstract Modeling base class
│   │   ├── classification.py            # Classification (LR, SVM, RF)
│   │   ├── survival.py                  # Survival analysis (CoxPH, CoxNet, IPC)
│   │   └── utils.py                     # Normalization (StandardScaler, MinMaxScaler)
│   ├── service/                         # Business logic layer
│   │   ├── feature_extraction.py        # Orchestrates extraction (Celery chord)
│   │   └── machine_learning.py          # Orchestrates ML training
│   ├── presets_default/                 # Default PyRadiomics YAML configs
│   ├── alembic/                         # Database migrations
│   └── scripts/                         # Data import/parsing scripts
├── workers/                             # Celery worker processes
│   ├── tasks.py                         # Task definitions (extract, train)
│   ├── utils.py                         # Worker utilities
│   ├── celeryconfig.py                  # Celery broker/backend config
│   └── config_workers.py               # Worker-specific Flask config
├── env_files/                           # Environment variable files
├── secrets/                             # Docker secrets (passwords)
├── docker-compose.yml                   # Base Docker Compose
├── docker-compose.override.yml          # Development overrides
├── docker-compose.local.yml             # Local deployment
└── docker-compose.prod.yml             # Production (Traefik, backups)
```

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| **Web Framework** | Flask | 1.1.2 |
| **ORM** | SQLAlchemy (via Flask-SQLAlchemy) | 1.3.24 |
| **Database** | MySQL | 5.7 |
| **Migrations** | Alembic | 1.14.1 |
| **Task Queue** | Celery | 5.5.0 |
| **Message Broker** | Redis | 7.4.1 |
| **WebSocket** | Flask-SocketIO (eventlet) | 4.3.2 |
| **Auth** | Keycloak (python-keycloak) | 3.6.1 |
| **DICOM** | pydicom, SimpleITK | 2.4.4, 2.4.1 |
| **Radiomics** | PyRadiomics (via Okapy) | 3.1.0 |
| **ML Classification** | scikit-learn | 1.3.2 |
| **ML Survival** | scikit-survival | 0.22.2 |
| **Data** | pandas, numpy | 1.5.3, 1.23.5 |
| **Model Persistence** | joblib | (bundled with sklearn) |
| **Code Formatter** | Black | 24.8.0 |

---

## Coding Standards

### Python Style

- **PEP 8** compliance, enforced by **Black** formatter (line length: Black default 88 chars).
- Use **type hints** for all new function signatures (parameters and return types).
- Use **f-strings** for string formatting, never `%` or `.format()`.
- Use **`pathlib.Path`** for file system operations in new code (existing code uses `os.path`).
- Prefer **list comprehensions** over `map()`/`filter()` for readability.
- Use **`logging`** module (not `print()`) for all new diagnostic output. Existing code uses `print()` — do not refactor unless explicitly asked.
- Follow existing naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.

### Type Hinting

```python
from typing import List, Dict, Optional, Tuple, Any

def process_features(
    extraction_id: int,
    feature_ids: List[str],
    normalize: bool = True,
) -> pd.DataFrame:
    ...
```

- Use `Optional[X]` for parameters that can be `None`.
- Use `-> None` for functions that return nothing.
- For SQLAlchemy models, type hint query results as the model class.

### Error Handling

```python
from quantimage2_backend_common.utils import InvalidUsage, ComputationError

# For client errors (bad request, missing data):
raise InvalidUsage("Patient labels are missing", status_code=400)

# For server-side computation errors:
raise ComputationError("Feature extraction failed for study {study_uid}", status_code=500)
```

- **Never swallow exceptions silently.** At minimum, log the error.
- Use specific exception types — avoid bare `except:`.
- For Celery tasks, catch `SoftTimeLimitExceeded` to handle timeout gracefully.
- Log all errors with `logging.error()` or `logging.exception()` (includes traceback).
- Medical data processing must fail loudly — never return partial results without explicit warning.

### Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Starting extraction {extraction_id} for album {album_id}")
logger.warning(f"Missing ROI {roi_name} in study {study_uid}")
logger.error(f"Extraction failed: {e}", exc_info=True)
```

- **NEVER log** patient names, JWT tokens, passwords, or Docker secrets.
- DICOM UIDs (StudyInstanceUID, SeriesInstanceUID) may be logged for debugging.
- Use structured context in log messages: include `extraction_id`, `study_uid`, `album_id` where applicable.

### Imports

```python
# 1. Standard library
import os
import logging
from typing import List, Dict

# 2. Third-party packages
import pandas as pd
import numpy as np
from flask import request, jsonify, g
from celery import Celery

# 3. Local/shared packages
from quantimage2_backend_common.models import FeatureExtraction, FeatureValue
from quantimage2_backend_common.utils import InvalidUsage
```

- Group imports in three sections: stdlib, third-party, local.
- Use absolute imports for the shared package: `from quantimage2_backend_common.X import Y`.
- Within webapp or workers, use relative imports for sibling modules: `from routes.utils import validate_decorate`.

---

## Database & ORM Patterns

### Model Definition

All models inherit from `BaseModel` (provides `id`, `created_at`, `updated_at`, CRUD methods):

```python
class MyModel(BaseModel, db.Model):
    def __init__(self, name: str, user_id: str):
        self.name = name
        self.user_id = user_id

    name = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.String(255), nullable=False)

    # Foreign keys
    extraction_id = db.Column(db.Integer, db.ForeignKey("feature_extraction.id"))

    # Methods
    @classmethod
    def find_by_user(cls, user_id: str) -> List["MyModel"]:
        return db.session.query(cls).filter_by(user_id=user_id).all()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
        }
```

### Key BaseModel Methods

| Method | Purpose |
|---|---|
| `find_by_id(id)` | Query by primary key |
| `delete_by_id(id)` | Delete + commit |
| `find_all()` | Query all records |
| `get_or_create(criteria, defaults)` | Upsert pattern with row-level locking |
| `save_to_db()` | Add + commit |
| `flush_to_db()` | Add + flush (no commit, for use within transactions) |
| `update(**kwargs)` | Update attributes + commit |

### Key ORM Models

| Model | Table | Purpose |
|---|---|---|
| `FeatureExtraction` | `feature_extraction` | One extraction run per album |
| `FeatureExtractionTask` | `feature_extraction_task` | One task per study in an extraction |
| `FeatureValue` | `feature_value` | Individual feature measurement |
| `FeatureDefinition` | `feature_definition` | Feature name registry |
| `Modality` | `modality` | CT, PET, MR, etc. |
| `ROI` | `roi` | GTV_T, GTV_N, etc. |
| `FeatureCollection` | `feature_collection` | User-defined feature subset |
| `Model` | `model` | Trained ML model record |
| `LabelCategory` | `label_category` | Outcome definition (Classification/Survival) |
| `Label` | `label` | Patient outcome values |
| `ClinicalFeatureDefinition` | `clinical_feature_definition` | Clinical feature schema |
| `ClinicalFeatureValue` | `clinical_feature_value` | Clinical feature data |

### Database Connection

```python
# Connection string pattern:
mysql://{DB_USER}:{secret}@db/{DB_DATABASE}

# Pool settings (flask_init.py):
SQLALCHEMY_POOL_SIZE = 10
SQLALCHEMY_MAX_OVERFLOW = 20
SQLALCHEMY_POOL_TIMEOUT = 30
SQLALCHEMY_POOL_PRE_PING = True  # Reconnect stale connections
SQLALCHEMY_POOL_RECYCLE = 1800   # Recycle every 30 minutes
```

### Migrations

```bash
# Generate a new migration:
cd webapp && alembic revision --autogenerate -m "Add new column"

# Apply migrations:
cd webapp && alembic upgrade head
```

The backend also supports auto-migration on startup via `DB_AUTOMIGRATE=1` environment variable.

---

## Authentication & Authorization

### How It Works

1. Frontend sends JWT token in `Authorization: Bearer <token>` header.
2. Every blueprint registers `@bp.before_request` → `validate_decorate(request)`.
3. `validate_decorate()` decodes the JWT using Keycloak's realm public key.
4. Sets `g.user` (Keycloak subject UUID) and `g.token` (raw JWT string).
5. Route handlers access the authenticated user via `g.user` and forward `g.token` to Kheops.

### Token Validation (routes/utils.py)

```python
from jose import jwt, JWTError, ExpiredSignatureError

def decode_token(token: str) -> dict:
    """Decode and validate a Keycloak JWT token."""
    global KEYCLOAK_REALM_PUBLIC_KEY
    if KEYCLOAK_REALM_PUBLIC_KEY is None:
        KEYCLOAK_REALM_PUBLIC_KEY = (
            "-----BEGIN PUBLIC KEY-----\n"
            + oidc_client.public_key()
            + "\n-----END PUBLIC KEY-----"
        )
    return jwt.decode(
        token,
        KEYCLOAK_REALM_PUBLIC_KEY,
        algorithms=["RS256"],
        options={"verify_signature": True, "verify_exp": True, "verify_aud": False},
    )
```

### Keycloak Configuration (config.py)

```python
oidc_client = KeycloakOpenID(
    server_url=os.environ["KEYCLOAK_BASE_URL"],            # http://keycloak:8081/auth/
    realm_name=os.environ["KEYCLOAK_REALM_NAME"],           # QuantImage-v2
    client_id=os.environ["KEYCLOAK_QUANTIMAGE2_FRONTEND_CLIENT_ID"],  # quantimage2-frontend
)
```

### Auth Patterns for New Routes

```python
from flask import Blueprint, g, request, jsonify
from routes.utils import validate_decorate, role_required

bp = Blueprint("my_routes", __name__)

@bp.before_request
def before_request():
    validate_decorate(request)

@bp.route("/my-endpoint/<int:item_id>", methods=["GET"])
def get_item(item_id: int):
    user_id = g.user    # Keycloak subject UUID
    token = g.token      # Raw JWT (forward to Kheops if needed)
    # ... business logic ...
    return jsonify(result)

@bp.route("/admin-endpoint", methods=["POST"])
@role_required("admin")
def admin_action():
    # Only accessible by users with "admin" role in Keycloak
    ...
```

### Rules

- **Every new route blueprint MUST include** `validate_decorate(request)` in `@bp.before_request`.
- Always scope DB queries by `g.user` (user_id) unless the endpoint is explicitly admin-only.
- Forward `g.token` to Kheops API calls — the same JWT is used for both backend and Kheops auth.
- **Never store tokens** in the database or log them.

---

## Kheops Integration (DICOMweb)

### API Client (kheops_utils.py)

```python
KHEOPS_BASE_URL = os.environ.get("KHEOPS_BASE_URL", "http://host.docker.internal")

class KheopsEndpoints:
    capabilities = f"{KHEOPS_BASE_URL}/api/capabilities"
    studies = f"{KHEOPS_BASE_URL}/api/studies"
    albums = f"{KHEOPS_BASE_URL}/api/albums"
    album_parameter = "album"
    seriesSuffix = "/series"
    instancesSuffix = "/instances"
    studyMetadataSuffix = "metadata"

def get_token_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
```

### Common API Calls

| Function | Method | URL | Purpose |
|---|---|---|---|
| `get_album_details()` | GET | `/api/albums/{album_id}` | Album metadata |
| `get_studies_from_album()` | GET | `/api/studies?album={album_id}` | List studies |
| `get_series_from_study()` | GET | `/api/studies/{study_uid}/series?album={album_id}&00080060={modality}` | Series by modality |
| `get_series_metadata()` | GET | `/api/studies/{study_uid}/series/{series_uid}/metadata?album={album_id}` | DICOM metadata |
| `download_study()` | GET | `/api/studies/{study_uid}?accept=application/zip&album={album_id}` | Download as ZIP |
| `get_user_token()` | POST | `/api/capabilities` | Create scoped capability token (24h, read+write) |

### Rules for Kheops Calls

- Always use `get_token_header(token)` for the `Authorization` header.
- For extraction tasks, create a **capability token** via `get_user_token(album_id, token)` — a time-limited scoped token.
- Handle HTTP errors explicitly — check `response.status_code` before calling `.json()`.
- DICOM field access uses hex tag codes: `DicomFields.STUDY_UID = "0020000D"`, `DicomFields.PATIENT_ID = "00100020"`, etc.
- **Never hardcode Kheops URLs** — always use `KHEOPS_BASE_URL` from environment.

---

## Celery Task Patterns

### Task Definitions (workers/tasks.py)

```python
from celery import Celery

app = Celery("quantimage2tasks")
app.config_from_envvar("CELERY_CONFIG_MODULE")

@app.task(
    bind=True,
    name="quantimage2tasks.extract",
    queue="extraction",
    time_limit=7200,
    soft_time_limit=6600,
)
def run_extraction(self, study_uid: str, album_id: str, ...):
    ...

@app.task(
    bind=True,
    name="quantimage2tasks.train",
    queue="training",
    serializer="pickle",
)
def train_model(self, ...):
    ...
```

### Task Queues

| Queue | Worker | Purpose | Concurrency |
|---|---|---|---|
| `extraction` | `celery_extraction` | Feature extraction (CPU/IO bound) | 2 |
| `training` | `celery_training` | ML model training (CPU bound) | 4 |

### Orchestration Pattern (Celery Chord)

Feature extraction uses a **chord** — parallel tasks with a final callback:

```python
from celery import chord

# One task per study, then finalize
extraction_chord = chord(
    [
        quantimage2tasks.extract.s(study_uid, album_id, ...)
        | finalize_extraction_task.s(task_id)
        for study_uid, task_id in studies_and_tasks
    ],
    finalize_extraction.si(extraction_id),
)
extraction_chord.apply_async()
```

### Progress Reporting

Tasks report progress via **Socket.IO** (through Redis message queue):

```python
from flask_socketio import SocketIO

socketio = SocketIO(message_queue=os.environ["SOCKET_MESSAGE_QUEUE"])

# Emit to all clients:
socketio.emit("extraction-status", status_dict)
socketio.emit("training-status", progress_dict)
```

### Socket.IO Event Types

| Event | Payload | Purpose |
|---|---|---|
| `extraction-status` | `{feature_extraction_id, ready, status, ...}` | Overall extraction progress |
| `feature-status` | `{task_id, status, ...}` | Per-study task status |
| `training-status` | `{training_id, phase, current, total, ...}` | Model training progress |

### Celery Configuration (celeryconfig.py)

```python
result_persistent = True
result_expires = None
task_track_started = True
task_acks_late = True
task_acks_on_failure_or_timeout = True
# Redis keepalive (detect broken connections):
broker_transport_options = {
    "socket_keepalive": True,
    "socket_keepalive_options": {TCP_KEEPIDLE: 60, TCP_KEEPINTVL: 10, TCP_KEEPCNT: 5},
    "health_check_interval": 30,
}
```

### Rules for New Tasks

- Always set `bind=True` to access `self` for state updates.
- Always specify `queue=` explicitly — never rely on the default queue.
- Set `time_limit` and `soft_time_limit` for extraction tasks.
- Use `serializer="pickle"` only for training tasks (scikit-learn objects).
- Report progress via Socket.IO, not just Celery state updates.
- Handle `SoftTimeLimitExceeded` gracefully (clean up temp files, update DB status).

---

## Feature Extraction Pipeline

### Flow

1. **Route**: POST `/extract/album/<album_id>` → `routes/features.py`
2. **Service**: `service/feature_extraction.py` → `run_feature_extraction()`
   - Creates `FeatureExtraction` + `FeatureExtractionTask` DB records
   - Saves YAML config to `/quantimage2-data/extractions/configs/{user_id}/{album_id}/`
   - Dispatches Celery chord to `extraction` queue
3. **Worker**: `workers/tasks.py` → `run_extraction()`
   - Downloads DICOM study from Kheops as ZIP
   - Extracts to temp directory
   - Runs `okapy.dicomconverter.converter.ExtractorConverter` with YAML config
   - Calls `store_features()` to persist to MySQL
   - Emits Socket.IO progress events
4. **Finalization**: Updates extraction status, emits final Socket.IO event

### Configuration Format

YAML files defining PyRadiomics extraction parameters. Default presets in `webapp/presets_default/`:
- `MRI.yaml`, `MRI_CT.yaml`, `MRI_PET.yaml`, `PETCT_all.yaml`

### Feature Storage

Features are stored in **MySQL** via `feature_storage.py`:
- DataFrame columns: `patient`, `modality`, `VOI` (ROI), `feature_name`, `feature_value`
- Creates/updates `Modality`, `ROI`, `FeatureDefinition` records
- Bulk-inserts `FeatureValue` records via `bulk_insert_mappings`

**HDF5 caching** for retrieval performance:
- Path: `/quantimage2-data/features-cache/extraction-{id}/features.h5`
- Format: pandas HDF5 `fixed` format
- Cache is invalidated/rebuilt when features change

### Feature ID Format

```
{modality}‑{roi}‑{feature_name}
```

**CRITICAL:** The separator is a **non-breaking hyphen** (`‑`, U+2011), NOT a regular hyphen (`-`). ROI names can contain regular hyphens (e.g., `GTV-T`). The regex `featureIDMatcher` in `const.py` parses this format.

### Feature Prefixes (by extractor)

| Source | Prefixes |
|---|---|
| PyRadiomics | `original`, `log`, `wavelet`, `gradient`, `square`, `squareroot`, `exponential`, `logarithm` |
| Riesz | `tex` |
| ZRAD | `zrad` |

### Rules for Feature Code

- Always use `FEATURE_ID_SEPARATOR` (U+2011) when constructing or parsing feature IDs.
- Store features via `store_features()` / `FeatureValue.save_features_batch()` — never insert directly.
- Clean up temporary DICOM files after extraction (use `try/finally` or context managers).
- Respect the YAML config format — validate configs before dispatching extraction.
- Feature names must start with a known prefix (PyRadiomics, Riesz, or ZRAD).

---

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

Clinical features are merged with radiomic features before training:
- Types: `Number`, `Categorical`
- Encodings: `None`, `One-Hot Encoding`, `Normalization`, `Ordered Categories`
- Missing values: `Drop`, `Mode`, `Median`, `Mean`, `None`

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

## Route & API Patterns

### Blueprint Structure

Every route file follows this pattern:

```python
from flask import Blueprint, request, jsonify, g
from routes.utils import validate_decorate

bp = Blueprint("route_name", __name__)

@bp.before_request
def before_request():
    validate_decorate(request)

@bp.route("/endpoint/<int:id>", methods=["GET"])
def get_endpoint(id: int):
    user_id = g.user
    # ... query scoped by user_id ...
    return jsonify(result)

@bp.route("/endpoint", methods=["POST"])
def create_endpoint():
    data = request.json
    user_id = g.user
    # ... validate, create, return ...
    return jsonify(result), 201
```

### Response Patterns

| Pattern | When to Use |
|---|---|
| `jsonify(data)` | Standard JSON responses |
| `jsonify(data), 201` | Created resources |
| `Response(content, mimetype="text/yaml")` | YAML file downloads |
| `Response(content, mimetype="application/zip")` | ZIP file downloads |
| `Response(content, mimetype="text/csv")` | CSV file downloads |
| `MultipartEncoder(fields={...})` | Multi-part responses (feature details) |

### Key Endpoints

| Method | Route | Purpose |
|---|---|---|
| POST | `/extract/album/<album_id>` | Trigger feature extraction |
| GET | `/extractions/album/<album_id>` | Latest extraction for album |
| GET | `/extractions/<id>/feature-details` | Features as multipart (tabular + chart) |
| GET | `/extractions/<id>/status` | Extraction progress |
| POST | `/models/<album_id>` | Train ML model |
| GET | `/models/<album_id>` | List models for album |
| POST | `/models/compare` | Compare two models (permutation test) |
| GET/PATCH | `/albums/<album_id>` | Album metadata + ROIs |
| GET | `/tasks/<task_id>/status` | Celery task status |

---

## Docker & Container Networking

### Services

| Service | Image | Port | Role |
|---|---|---|---|
| `backend` | `quantimage2_backend:3.1` | 5000 | Flask API + Socket.IO |
| `celery_extraction` | `quantimage2_backend_celery:3.1` | — | Feature extraction workers |
| `celery_training` | `quantimage2_backend_celery:3.1` | — | ML training workers |
| `flower` | `quantimage2_flower:3.0` | 3333 | Celery monitoring |
| `db` | `mysql:5.7` | 3306 | MySQL database |
| `redis` | `redis:7.4.1` | 6379 | Celery broker + result backend |
| `redis-socket` | `redis:7.4.1` | 6379 | Socket.IO message queue |
| `keycloak` | `keycloak:16.1.1` | 8081 | Auth server (dev profile) |
| `keycloak-db` | `postgres:17.6` | — | Keycloak database (dev profile) |
| `phpmyadmin` | `phpmyadmin` | 8888 | Database admin UI |
| `ohif` | `ohif/viewer:v4.12.35.19437` | 4567 | DICOM viewer |

### Inter-Service Communication

| From | To | Address | Protocol |
|---|---|---|---|
| backend, workers | `db` | `db:3306` | MySQL |
| backend, workers | `redis` | `redis:6379` | Redis (Celery) |
| backend, workers | `redis-socket` | `redis-socket:6379` | Redis (Socket.IO) |
| backend, workers | Kheops | `KHEOPS_BASE_URL` (host.docker.internal) | HTTP |
| backend | `keycloak` | `keycloak:8081` | HTTP |
| workers | `flower` | `flower:3333` | HTTP (task result API) |
| frontend (browser) | `backend` | `localhost:5000` | HTTP + WebSocket |

### Networking Rules

- **Default bridge network** — all services communicate by Docker service name.
- In development, `extra_hosts` maps `host.docker.internal` → host gateway (for Kheops running on host).
- In production, a **Traefik** external network (`${TRAEFIK_NETWORK}`) is added for reverse proxy.
- **Never hardcode container IPs** — always use service names or environment variables.
- Frontend calls go **from the browser**, not between containers — so `CORS_ALLOWED_ORIGINS` must match.

### Shared Volume

```
backend-data:/quantimage2-data
```

Used by `backend`, `celery_extraction`, and `celery_training` for:
- Feature extraction configs: `/quantimage2-data/extractions/`
- Feature cache (HDF5): `/quantimage2-data/features-cache/`
- Trained models (joblib): `/quantimage2-data/models/`
- Feature preset uploads: `/quantimage2-data/feature-presets/`

### Rules for Docker Changes

- Backend, `celery_extraction`, and `celery_training` share the same `backend-data` volume.
- New services must be added to the base `docker-compose.yml` and all overlay files.
- Use Docker **secrets** for passwords — never put credentials in env files.
- Production deployments use Traefik labels for routing — follow existing label patterns.
- Health checks are mandatory for services other containers depend on (e.g., `db`).

---

## Environment Variables

### Common (env_files/common.env)

| Variable | Value | Purpose |
|---|---|---|
| `DB_DATABASE` | `quantimage2` | MySQL database name |
| `DB_USER` | `quantimage2` | MySQL username |
| `KEYCLOAK_BASE_URL` | `http://keycloak:8081/auth/` | Keycloak server URL |
| `KEYCLOAK_REALM_NAME` | `QuantImage-v2` | Keycloak realm |
| `KEYCLOAK_QUANTIMAGE2_FRONTEND_CLIENT_ID` | `quantimage2-frontend` | OIDC client ID |
| `KEYCLOAK_FRONTEND_ADMIN_ROLE` | `admin` | Admin role name |
| `KHEOPS_BASE_URL` | `http://host.docker.internal` | Kheops PACS base URL |
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Celery result store |
| `SOCKET_MESSAGE_QUEUE` | `redis://redis-socket:6379/` | Socket.IO message queue |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000` | Frontend origin(s) |

### Webapp (env_files/webapp.env)

| Variable | Value | Purpose |
|---|---|---|
| `GRID_SEARCH_CONCURRENCY` | `-1` | GridSearchCV parallelism (all cores) |

### Workers (env_files/workers.env)

| Variable | Value | Purpose |
|---|---|---|
| `CELERY_WORKER_CONCURRENCY_EXTRACTION` | `2` | Extraction worker concurrency |
| `CELERY_WORKER_CONCURRENCY_TRAINING` | `4` | Training worker concurrency |
| `CELERY_CONFIG_MODULE` | `celeryconfig` | Celery config module |
| `C_FORCE_ROOT` | `true` | Allow Celery to run as root |

### Secrets

| Secret | File | Purpose |
|---|---|---|
| `db-root-password` | `secrets/db_root_password` | MySQL root password |
| `db-user-password` | `secrets/db_user_password` | MySQL user password |
| `keycloak-admin-password` | `secrets/keycloak_admin_password` | Keycloak admin |
| `keycloak-db-user-password` | `secrets/keycloak_db_user_password` | Keycloak DB password |
| `quantimage2-admin-password` | `secrets/quantimage2_admin_password` | App admin password |

---

## Key Constants (const.py)

### Enums

```python
class MODEL_TYPES(Enum):
    CLASSIFICATION = "Classification"
    SURVIVAL = "Survival"

class DATA_SPLITTING_TYPES(Enum):
    FULLDATASET = "fulldataset"
    TRAINTESTSPLIT = "traintest"

class TRAIN_TEST_SPLIT_TYPES(Enum):
    AUTO = "automatic"
    MANUAL = "manual"

class ESTIMATOR_STEP(Enum):
    CLASSIFICATION = "classifier"
    SURVIVAL = "analyzer"

class TRAINING_PHASES(Enum):
    TRAINING = "training"
    TESTING = "testing"
```

### Task Queues

```python
QUEUE_EXTRACTION = "extraction"
QUEUE_TRAINING = "training"
```

### Feature Format

```python
FEATURE_ID_SEPARATOR = "\u2011"  # Non-breaking hyphen (U+2011)
CV_SPLITS = 5
CV_REPEATS = 1
```

---

## Security & Medical Data Handling

### Rules

- **NEVER** log patient names, DICOM PatientName/PatientID, JWT tokens, or passwords.
- DICOM UIDs may be logged for debugging (they are not PHI in most contexts).
- All user data must be scoped by `g.user` (Keycloak subject UUID) — never expose other users' data.
- Feature extraction YAML configs may contain PHI-adjacent information (ROI names) — treat as sensitive.
- Temporary DICOM files must be deleted after extraction — use `try/finally` blocks.
- Model files on disk are scoped by `{user_id}/{album_id}/` paths.
- Docker secrets for all passwords — never commit credentials to the repository.
- Input validation: validate all user-supplied IDs, file paths, and config data before processing.
- Use `pathvalidate` for filename validation when handling user-uploaded files.

---

## Frontend Contract

The React frontend communicates with this backend via:

1. **REST API** (all endpoints require `Authorization: Bearer <token>` header)
2. **Socket.IO** (WebSocket, via eventlet, through `redis-socket` message queue)

### Frontend Expectations

- JSON responses with consistent structure (use `to_dict()` methods on models).
- Multipart responses for feature details (tabular + chart CSV).
- Socket.IO events for real-time progress (extraction, training).
- HTTP status codes: 200 (OK), 201 (Created), 400 (Bad Request via `InvalidUsage`), 500 (Server Error via `ComputationError`).
- CORS headers set via `Flask-CORS` with `CORS_ALLOWED_ORIGINS`.

---

## Code Formatting

After every code edit, ensure modified Python files conform to **Black** formatting:

```bash
black <modified_files>
```

Verify:
```bash
black --check <modified_files>
```

Format only the files you changed — do not reformat the entire codebase.

---

## Testing & Validation

### Running the Stack

```bash
# Development (with source code mounts):
docker-compose up

# Local deployment (built images):
docker-compose -f docker-compose.yml -f docker-compose.local.yml up

# Production:
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Validate Docker Compose

```bash
docker-compose config --quiet && echo "Config OK"
```

### Database Access

```bash
# Via phpMyAdmin:
http://localhost:8888

# Via MySQL CLI:
docker-compose exec db mysql -u quantimage2 -p quantimage2
```

### Celery Monitoring

```bash
# Flower dashboard:
http://localhost:3333

# Check worker status:
docker-compose exec celery_extraction celery -A tasks inspect active
```

---

## Workflow Prompts

Reusable prompt templates are available in `.github/prompts/` for common workflows:

| Prompt | Purpose |
|---|---|
| `radiomics-feature-engineering` | Creating/modifying feature extraction scripts with PyRadiomics/Okapy |
| `ml-pipeline` | Extending the ML training pipeline (new algorithms, metrics, preprocessing) |
| `integration-api` | Building new API endpoints with Keycloak auth and Kheops integration |
| `build-validator` | Validating Docker builds, compose configs, and deployment readiness |
| `code-architect` | Reviewing code for architecture consistency and patterns |
| `oncall-guide` | Production troubleshooting reference |

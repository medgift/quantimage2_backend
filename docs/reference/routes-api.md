# Route & API Patterns + Frontend Contract

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

**Clinical features** (multi-CSV; `album_id` is a query param, and feature IDs are namespaced `{file_id}::{name}`):

| Method | Route | Purpose |
|---|---|---|
| GET/POST | `/clinical-features-files` | List / create a clinical-features file (server uniquifies the name) for an album |
| PATCH/DELETE | `/clinical-features-files/<id>` | Rename (409 on name clash) / delete a file (cascades to its definitions + values) |
| GET/POST/PATCH | `/clinical-features-definitions` | List / save (requires `clinical_feature_file_id`) / update (ownership-checked; 400/404 guards) |
| DELETE | `/clinical-features-definitions` | **Disabled — returns 410.** Per-album wipe removed; delete a specific file instead |
| POST | `/clinical-features-definitions/guess` | Guess type / encoding / missing-value handling per column |
| POST | `/clinical-features` | Save values (`clinical_feature_map` + `clinical_feature_file_id`; replaces, not appends) **or** read values (`patient_ids` → `{patient_id: {file_id::name: value}}`) |
| POST | `/clinical-features/unique-values` | Value distribution per column (skips columns with no definition) |
| POST | `/clinical-features/filter` | Detect unusable columns (dates, <10% populated, single-value) |

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


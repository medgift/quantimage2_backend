# Integration & API — New Endpoints with Keycloak Auth and Kheops Integration

Use this prompt when building new REST API endpoints that require authentication, Kheops PACS interactions, or both.

## Context

@workspace This backend uses Flask Blueprints with Keycloak JWT authentication and Kheops DICOMweb API integration.

## Authentication Pattern

Every route blueprint **must** validate the JWT token:

```python
from flask import Blueprint, request, jsonify, g
from routes.utils import validate_decorate, role_required

bp = Blueprint("my_routes", __name__)

@bp.before_request
def before_request():
    validate_decorate(request)

@bp.route("/my-endpoint/<int:item_id>", methods=["GET"])
def get_item(item_id: int):
    user_id = g.user    # Keycloak subject UUID
    token = g.token      # Raw JWT (forward to Kheops)
    # ... business logic ...
    return jsonify(result)
```

**`validate_decorate(request)`** does:
1. Skips OPTIONS requests (CORS preflight)
2. Extracts Bearer token from `Authorization` header
3. Decodes JWT using Keycloak realm public key (RS256, verify_signature=True, verify_exp=True, verify_aud=False)
4. Sets `g.user` = Keycloak `sub` claim (user UUID)
5. Sets `g.token` = raw JWT string

**Admin-only endpoints** use the `@role_required("admin")` decorator.

## Kheops Integration Pattern

```python
from quantimage2_backend_common.kheops_utils import (
    KheopsEndpoints,
    get_token_header,
    get_album_details,
    get_studies_from_album,
    get_series_from_study,
    get_series_metadata,
    get_user_token,
    dicomFields as DicomFields,
)

# GET album details
album = get_album_details(album_id, token)

# GET studies from album
studies = get_studies_from_album(album_id, token)

# GET series from study (filtered by modality)
series = get_series_from_study(study_uid, album_id, modality, token)

# Custom request to Kheops
response = requests.get(
    f"{KheopsEndpoints.studies}/{study_uid}",
    params={KheopsEndpoints.album_parameter: album_id},
    headers=get_token_header(token),
)

# For extraction tasks, create a capability token (time-limited, scoped)
capability_token = get_user_token(album_id, token)
```

**Rules**:
- Always use `get_token_header(token)` for Authorization header
- Never hardcode Kheops URLs — use `KHEOPS_BASE_URL` from environment
- Forward `g.token` to Kheops calls (same JWT for both backend and Kheops)
- DICOM fields use hex tag codes: `DicomFields.STUDY_UID = "0020000D"`, `DicomFields.PATIENT_ID = "00100020"`, etc.
- Check `response.status_code` before calling `.json()`

## Blueprint Registration

After creating a new blueprint, register it in `webapp/app.py`:

```python
from routes.my_routes import bp as my_routes_bp

# Inside setup_app():
app.register_blueprint(my_routes_bp)
```

## Response Patterns

| Pattern | When |
|---|---|
| `return jsonify(data)` | Standard JSON |
| `return jsonify(data), 201` | Created resource |
| `return Response(content, mimetype="text/yaml")` | YAML download |
| `return Response(content, mimetype="application/zip")` | ZIP download |
| `return Response(content, mimetype="text/csv")` | CSV download |
| `MultipartEncoder(fields={...})` | Multi-part (feature details) |

## Error Handling

```python
from quantimage2_backend_common.utils import InvalidUsage, ComputationError

# Client errors (400):
raise InvalidUsage("Missing required field 'name'", status_code=400)

# Server errors (500):
raise ComputationError("Failed to process study {study_uid}", status_code=500)
```

The Flask app has a registered `@errorhandler(InvalidUsage)` that converts these to JSON responses.

## DB Queries

Always scope queries by `g.user`:

```python
models = Model.query.filter_by(user_id=g.user, album_id=album_id).all()
```

Use model `to_dict()` methods for JSON serialization. Use `BaseModel` methods: `find_by_id()`, `save_to_db()`, `update()`, `delete_by_id()`, `get_or_create()`.

## Key Files

- `webapp/routes/utils.py` — `validate_decorate()`, `decode_token()`, `role_required()`
- `webapp/config.py` — Keycloak OIDC client configuration
- `shared/quantimage2_backend_common/kheops_utils.py` — Kheops API client, `DicomFields`
- `shared/quantimage2_backend_common/utils.py` — `InvalidUsage`, `ComputationError`
- `shared/quantimage2_backend_common/models.py` — All ORM models with `to_dict()` methods
- `webapp/app.py` — Blueprint registration in `setup_app()`

## Checklist for New Endpoints

1. [ ] Create blueprint with `@bp.before_request` → `validate_decorate(request)`
2. [ ] Use `g.user` for user scoping, `g.token` for Kheops API calls
3. [ ] Add type hints to route handler parameters and return types
4. [ ] Use `InvalidUsage` for 400 errors, `ComputationError` for 500 errors
5. [ ] Return JSON via `jsonify()` with appropriate HTTP status codes
6. [ ] Register blueprint in `webapp/app.py` → `setup_app()`
7. [ ] Scope all DB queries by `g.user` (unless admin-only)
8. [ ] Never log tokens, passwords, or patient names
9. [ ] Run `black` on modified files
10. [ ] If new DB columns needed, create Alembic migration

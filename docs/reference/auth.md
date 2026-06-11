# Authentication, Authorization & Medical-Data Handling

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


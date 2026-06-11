# Environment Variables, Secrets & Key Constants

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


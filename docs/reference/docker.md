# Docker & Container Networking

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


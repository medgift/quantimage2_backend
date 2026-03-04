# Build Validator — Docker Build & Deployment Validation

Run this prompt to validate Docker builds, configurations, and deployment readiness for the backend.

## When to Use

- After modifying `Dockerfile`, `docker-compose*.yml`, or environment files
- After adding/removing Python packages in `requirements.txt`
- After changing environment variables or Docker secrets
- Before deploying to production
- After modifying `entrypoint.sh` or Celery worker commands

## Checks to Perform

### 1. Docker Compose Validation

```bash
cd /srv/quantimage-v2/quantimage2_backend

# Base config
docker compose config --quiet && echo "Base config OK"

# Development overlay
docker compose -f docker-compose.yml -f docker-compose.override.yml config --quiet && echo "Dev config OK"

# Local deployment
docker compose -f docker-compose.yml -f docker-compose.local.yml config --quiet && echo "Local config OK"

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --quiet && echo "Prod config OK"
```

### 2. Docker Build (Backend)

```bash
docker compose build backend 2>&1 | tail -15
docker compose build celery_extraction 2>&1 | tail -15
```

Both images must build successfully. The backend and workers share the same base (`shared/` package installed via `setup.py`).

### 3. Environment Variables

Verify all required variables are defined:

```bash
# Common env
cat env_files/common.env

# Required vars:
# DB_DATABASE, DB_USER, KEYCLOAK_BASE_URL, KEYCLOAK_REALM_NAME,
# KEYCLOAK_QUANTIMAGE2_FRONTEND_CLIENT_ID, KHEOPS_BASE_URL,
# CELERY_BROKER_URL, CELERY_RESULT_BACKEND, SOCKET_MESSAGE_QUEUE,
# CORS_ALLOWED_ORIGINS

# Webapp env
cat env_files/webapp.env

# Workers env
cat env_files/workers.env
```

### 4. Docker Secrets

```bash
# All secret files must exist and be non-empty
for f in secrets/db_root_password secrets/db_user_password secrets/keycloak_admin_password secrets/keycloak_db_user_password secrets/quantimage2_admin_password; do
  [ -s "$f" ] && echo "✓ $f" || echo "✗ $f MISSING/EMPTY"
done
```

### 5. Python Dependencies

```bash
# Check requirements files for syntax errors
pip install --dry-run -r webapp/requirements.txt 2>&1 | tail -5
pip install --dry-run -r workers/requirements.txt 2>&1 | tail -5
```

### 6. Shared Package

```bash
# Verify setup.py is valid
cd shared && python setup.py check 2>&1
```

### 7. Alembic Migrations

```bash
# Check migration chain is valid (no branch conflicts)
cd webapp && alembic check 2>&1
```

### 8. Services Start

```bash
# Start all services
docker compose up -d

# Check service health
docker compose ps

# Backend responds
curl -sf http://localhost:5000/ && echo "Backend OK"

# Flower responds
curl -sf http://localhost:3333/ && echo "Flower OK"

# DB healthy
docker compose exec db mysqladmin ping -u root --password=$(cat secrets/db_root_password) 2>&1
```

### 9. Celery Workers

```bash
# Check workers are registered
docker compose exec celery_extraction celery -A tasks inspect active 2>&1 | head -10
docker compose exec celery_training celery -A tasks inspect active 2>&1 | head -10
```

### 10. Redis Connectivity

```bash
docker compose exec redis redis-cli ping
docker compose exec redis-socket redis-cli ping
```

### 11. Code Formatting

```bash
# Verify Black formatting on modified files
black --check webapp/ workers/ shared/ 2>&1
```

## Report Format

```
✓ Docker compose configs: valid
✓ Docker builds: successful
✓ Environment variables: complete
✓ Secrets: present
✓ Python deps: resolvable
✓ Shared package: valid
✓ Alembic: no conflicts
✓ Services: running
✓ Celery workers: connected
✓ Redis: responsive
✓ Code formatting: compliant
```

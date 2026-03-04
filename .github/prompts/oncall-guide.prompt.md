# On-Call Guide — Production Troubleshooting Reference

Use this prompt for diagnosing production issues with the QuantImage v2 backend.

## Context

@workspace The backend runs as multiple Docker containers: `backend` (Flask API), `celery_extraction`, `celery_training`, `flower`, `db` (MySQL), `redis`, `redis-socket`. External services: Kheops (PACS), Keycloak (auth).

## Quick Diagnostics

### Service Status

```bash
cd /srv/quantimage-v2/quantimage2_backend
docker compose ps
docker compose logs --tail=50 backend
docker compose logs --tail=50 celery_extraction
docker compose logs --tail=50 celery_training
```

### Health Checks

```bash
# Backend API
curl -sf http://localhost:5000/ && echo "Backend OK" || echo "Backend DOWN"

# MySQL
docker compose exec db mysqladmin ping -u root --password=$(cat secrets/db_root_password)

# Redis (Celery broker)
docker compose exec redis redis-cli ping

# Redis (Socket.IO)
docker compose exec redis-socket redis-cli ping

# Flower (Celery monitoring)
curl -sf http://localhost:3333/ && echo "Flower OK" || echo "Flower DOWN"

# Keycloak
curl -sf http://localhost:8081/auth/realms/QuantImage-v2/.well-known/openid-configuration | head -1
```

## Common Issues

### 1. Feature Extraction Stuck

**Symptoms**: Extraction shows "in progress" indefinitely, no Socket.IO updates.

**Diagnosis**:
```bash
# Check extraction workers
docker compose exec celery_extraction celery -A tasks inspect active

# Check Redis for stuck tasks
docker compose exec redis redis-cli keys "celery-task-meta-*" | wc -l

# Check task state
docker compose exec redis redis-cli GET "celery-task-meta-<task-id>"

# Check for zombie processes
docker compose exec celery_extraction ps aux
```

**Resolution**:
- Restart extraction worker: `docker compose restart celery_extraction`
- Cancel stuck extraction via API: POST `/extractions/<id>/cancel`
- Check Kheops connectivity: `docker compose exec backend curl -sf $KHEOPS_BASE_URL/api`
- Check disk space for DICOM temp files: `docker compose exec celery_extraction df -h /tmp`

### 2. Model Training Failures

**Symptoms**: Training fails, Socket.IO reports error, or training hangs.

**Diagnosis**:
```bash
# Check training worker logs
docker compose logs --tail=100 celery_training

# Check active training tasks
docker compose exec celery_training celery -A tasks inspect active

# Check Flower for task details
curl http://localhost:3333/api/tasks?state=FAILURE | python -m json.tool
```

**Common causes**:
- Too few samples for CV folds (need ≥5 per class, or min class size samples).
- All features constant → scaler fails.
- Survival data with no events → C-index undefined.
- Memory exhaustion → OOM kill (check `docker stats`).

### 3. Database Connection Errors

**Symptoms**: `OperationalError: (2006, 'MySQL server has gone away')` or `Can't connect to MySQL server`.

**Diagnosis**:
```bash
# Check DB container
docker compose ps db
docker compose logs --tail=20 db

# Check connection pool
docker compose exec db mysql -u quantimage2 -p -e "SHOW PROCESSLIST"

# Check pool settings
grep -i pool shared/quantimage2_backend_common/flask_init.py
```

**Resolution**:
- Pool settings: `pool_pre_ping=True` (auto-reconnect), `pool_recycle=1800` (30min)
- Restart backend: `docker compose restart backend`
- Check max_connections: `docker compose exec db mysql -u root -p -e "SHOW VARIABLES LIKE 'max_connections'"`

### 4. Socket.IO Not Working

**Symptoms**: Frontend shows no real-time updates for extraction/training progress.

**Diagnosis**:
```bash
# Check redis-socket
docker compose exec redis-socket redis-cli ping

# Check backend Socket.IO logs
docker compose logs --tail=50 backend | grep -i socket

# Test Socket.IO transport
curl -sf "http://localhost:5000/socket.io/?transport=polling"
```

**Resolution**:
- Restart redis-socket: `docker compose restart redis-socket`
- Check CORS: `grep CORS_ALLOWED_ORIGINS env_files/common.env`
- Restart backend: `docker compose restart backend`

### 5. Keycloak Token Validation Failures

**Symptoms**: All API calls return 401 Unauthorized.

**Diagnosis**:
```bash
# Check Keycloak container
docker compose ps keycloak

# Check realm exists
curl -sf http://localhost:8081/auth/realms/QuantImage-v2 | head -5

# Check public key cache in backend
docker compose logs --tail=20 backend | grep -i keycloak
```

**Resolution**:
- Restart backend to refresh public key cache
- Check `KEYCLOAK_BASE_URL` in `env_files/common.env`
- Verify Keycloak realm name matches `KEYCLOAK_REALM_NAME`

### 6. Kheops Connectivity Issues

**Symptoms**: Extraction fails with connection errors to Kheops.

**Diagnosis**:
```bash
# Test Kheops from backend container
docker compose exec backend curl -sf $KHEOPS_BASE_URL/api

# Check host.docker.internal resolution
docker compose exec backend getent hosts host.docker.internal
```

**Resolution**:
- Verify `KHEOPS_BASE_URL` in `env_files/common.env`
- In dev, ensure `extra_hosts` mapping is present in `docker-compose.override.yml`
- Check Kheops service is running on the host machine

### 7. Disk Space Issues

**Symptoms**: Extraction fails with "No space left on device".

**Check**:
```bash
# Shared volume usage
docker compose exec backend du -sh /quantimage2-data/*

# Temp DICOM files
docker compose exec celery_extraction du -sh /tmp/quantimage*

# Docker system
docker system df
```

**Resolution**:
- Clean old feature caches: `/quantimage2-data/features-cache/`
- Clean old model files: `/quantimage2-data/models/`
- Docker prune: `docker system prune -f`

## Log Locations

| Service | Command |
|---|---|
| Backend (Flask) | `docker compose logs backend` |
| Extraction Worker | `docker compose logs celery_extraction` |
| Training Worker | `docker compose logs celery_training` |
| MySQL | `docker compose logs db` |
| Redis | `docker compose logs redis` |
| Flower | `docker compose logs flower` / `http://localhost:3333` |

## Key Configuration

| Setting | Location | Value |
|---|---|---|
| DB pool size | `flask_init.py` | 10 (max overflow 20) |
| Extraction concurrency | `workers-extraction.env` | 2 |
| Training concurrency | `workers-training.env` | 4 |
| Extraction timeout | `workers/tasks.py` | 7200s hard, 6600s soft |
| Redis keepalive | `celeryconfig.py` | 60s idle, 10s interval, 5 retries |

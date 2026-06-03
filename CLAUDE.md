# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It stays focused on commands, architecture-at-a-glance, and gotchas not obvious from reading code.

## Reference docs (read on demand)

Long-form detail lives in [`docs/reference/`](docs/reference/README.md), split by topic so you only load what a task needs — **don't** inline these into this file. Read the relevant one when you need depth; don't re-derive it from the codebase:

- [`architecture.md`](docs/reference/architecture.md) — project overview, data flow, repository structure, tech-stack versions
- [`coding-standards.md`](docs/reference/coding-standards.md) — Python style, typing, error handling, logging, imports, formatting, testing
- [`database.md`](docs/reference/database.md) — ORM/model patterns, BaseModel helpers, connection/pool, migrations
- [`auth.md`](docs/reference/auth.md) — Keycloak OIDC, token validation, per-route auth, medical-data handling
- [`kheops.md`](docs/reference/kheops.md) — Kheops DICOMweb client and call patterns
- [`celery.md`](docs/reference/celery.md) — task definitions, queues, chord orchestration, Socket.IO progress, celeryconfig
- [`feature-extraction.md`](docs/reference/feature-extraction.md) — extraction flow, config format, feature storage, feature-ID format & prefixes
- [`ml-modeling.md`](docs/reference/ml-modeling.md) — classification/survival algorithms, training flow, pipeline, clinical features
- [`routes-api.md`](docs/reference/routes-api.md) — blueprint/route patterns, response shapes, key endpoints, frontend contract
- [`docker.md`](docs/reference/docker.md) — services, inter-service networking, shared volume
- [`config.md`](docs/reference/config.md) — environment variables, secrets, key constants

These were recovered from the former `.github/copilot-instructions.md`; treat them as reference, not contract — verify specifics against the code.

## Repository scope

This is the **backend** repo of the QuantImage v2 platform — a radiomics research platform for medical imaging. Sibling repos (cloned by `quantimage2-setup`):
- `quantimage2-frontend` — React SPA (talks to this backend over REST + Socket.IO)
- `quantimage2-kheops` — DICOMweb PACS configuration
- `okapy` — radiomics extraction wrapper around PyRadiomics

This repo is normally checked out under `../quantimage2-setup/quantimage2_backend/` so the parent directory contains setup scripts and the data-mount directory (`quantimage2-data/`).

## High-level architecture

Three Python processes run in containers, sharing a MySQL DB, two Redis instances, and a host-mounted volume:

```
React SPA ──HTTP+WebSocket──> backend (Flask + Socket.IO, eventlet, :5000)
                                  │
                                  ├──SQLAlchemy──> db (MySQL 5.7)
                                  ├──Celery────── > redis (broker/result)
                                  ├──Socket.IO──> redis-socket (msg queue)
                                  ├──DICOMweb──> Kheops (host.docker.internal)
                                  └──OIDC──────> keycloak (dev profile only)

celery_extraction worker ── extraction queue ── PyRadiomics/Okapy ──> MySQL feature rows
celery_training   worker ── training   queue ── scikit-learn / scikit-survival ──> joblib model files
                                  │
                                  └──Socket.IO emit──> redis-socket ──> backend ──> SPA (live progress)
```

**`shared/quantimage2_backend_common`** is a local Python package imported by both `webapp/` and `workers/`. Models, DB init, Kheops client, Socket.IO helpers, and feature-storage logic live there so the same code paths run in both processes. In Docker it's mounted directly into `site-packages/` (see `docker-compose.override.yml`); for local Python work, install it editable.

**Feature ID format** — `{modality}‑{roi}‑{feature_name}` where the separator is the **non-breaking hyphen U+2011** (`FEATURE_ID_SEPARATOR` in `shared/quantimage2_backend_common/const.py`), not a regular `-`. ROI names themselves can contain regular hyphens (e.g. `GTV-T`). Always parse via the regex/constants in `const.py`; never split on `-`. The `{feature_name}` must begin with a known prefix from `prefixes` (consumed by `featureIDMatcher`). **`ZRAD_FEATURE_PREFIXES = ["zrad"]` is kept for parsing only** — ZRAD features are no longer computed (its build/extraction support was removed), but the prefix must stay so existing ZRAD feature IDs already in the DB still parse. Removing it makes `featureIDMatcher` silently drop or misclassify those rows (see `models.py` call sites). Same reasoning applies to `RIESZ_FEATURE_PREFIXES = ["tex"]`.

**Estimator step naming differs by model type**: `Pipeline` step name is `"classifier"` for classification and `"analyzer"` for survival (see `ESTIMATOR_STEP` enum). New ML code must respect this when poking pipeline internals.

**The `backend-data` Docker volume** (`/quantimage2-data` inside containers) is shared by `backend`, `celery_extraction`, and `celery_training`. Anything written there must use a path scoped by `{user_id}/{album_id}/`. The host-side mount path is set by `QUANTIMAGE2_DATA_MOUNT_DIRECTORY` in `.env`.

## Common commands

### Run the full stack

```bash
# Development (mounts source code, exposes ports 5000/3307/8888/8081/4567/3333/6379/6380):
docker compose up

# With dev Keycloak + Postgres:
docker compose --profile dev up

# Local deployment (built images, no source mount):
docker compose -f docker-compose.yml -f docker-compose.local.yml up

# Production (Traefik, backups):
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Validate a compose config without starting anything:
docker compose config --quiet && echo "Config OK"
```

The override file requires a root-level `.env` defining `QUANTIMAGE2_DATA_MOUNT_DIRECTORY` (see `.env.sample` for the production-shaped example; for local dev it just needs the data-mount line).

### Tests

`pytest.ini` lives at the repo root; tests are in `tests/`. `conftest.py` stubs Keycloak, swaps in an in-memory SQLite DB, mocks Celery + Socket.IO, and inserts `shared/`, `webapp/`, `workers/` onto `sys.path` so imports resolve outside Docker too.

```bash
# Inside a Python env that has tests/requirements-test.txt + webapp/requirements.txt installed:
pytest                                     # full suite
pytest -m unit                             # only unit-marked tests
pytest tests/test_ml_pipeline.py           # one file
pytest tests/test_models.py::TestFeatureExtraction::test_save  # one test
pytest -k "feature_storage and not slow"   # by keyword
pytest --cov=shared --cov=webapp           # with coverage

# Inside the running backend container:
docker compose exec backend pytest
```

Markers: `unit`, `integration`, `slow`, `ml` (declared in `pytest.ini`).

### Database

```bash
# MySQL CLI (password is in secrets/db_user_password):
docker compose exec db mysql -u quantimage2 -p quantimage2

# phpMyAdmin: http://localhost:8888

# Generate a new Alembic migration (run inside the backend container, where alembic.ini lives):
docker compose exec backend alembic revision --autogenerate -m "describe change"

# Apply migrations:
docker compose exec backend alembic upgrade head
```

**Migration workflow (read this before changing a model).** On startup the backend `entrypoint.sh` runs **only** `alembic upgrade head` when `DB_AUTOMIGRATE=1` (set in `env_files/debug.env` for dev, `docker-compose.local.yml` for local). It does **not** auto-generate migrations — that was removed because it created an empty revision file on every boot. So:

- **When you change a model** (`shared/quantimage2_backend_common/models.py`), you must generate the migration yourself, review it, and apply it:
  ```bash
  docker compose exec backend alembic revision --autogenerate -m "describe change"
  # review the generated file; autogenerate only emits schema DDL — add any data
  # backfill by hand (e.g. populating a new NOT NULL column for existing rows)
  docker compose exec backend alembic upgrade head
  ```
- **Migration files are git-ignored** (`.gitignore`: `webapp/alembic/versions/*`), so they do **not** travel to other machines or prod via git. A migration that matters (especially one with a hand-written data backfill) must be **force-added** (`git add -f <file>`) or applied to the target DB by hand — otherwise it exists only on the machine that generated it.
- **Prod** does not set `DB_AUTOMIGRATE`, so it never runs alembic on boot. Apply migrations there manually after deploy: `docker compose -f docker-compose.yml -f docker-compose.prod.yml exec backend alembic upgrade head` (take a DB backup first). This only works if the migration file is physically present on the prod server — see the git-ignore note above.

### Formatting

Black (24.8.0) is the canonical formatter — line length default (88). Format only the files you changed:

```bash
black <files>
black --check <files>
```

### Celery monitoring

```bash
# Flower UI: http://localhost:3333
docker compose exec celery_extraction celery -A tasks inspect active
docker compose logs -f celery_extraction celery_training
```

## Gotchas worth flagging up front

- **`eventlet.monkey_patch()` runs at the very top of `webapp/app.py`** — must stay there. Any import before it that touches sockets or threads will misbehave.
- **Worker code runs as a Flask app context too**, but uses `workers/config_workers.py` instead of `webapp/config.py`. Do not assume Flask config from the webapp is available inside Celery tasks.
- **`get_or_create()` uses row-level locking** (`SELECT ... FOR UPDATE`) — fine for the common upsert paths but it can deadlock with parallel extraction workers if you call it on a table that another task is also writing. When extending feature storage, prefer the existing batch insert helpers.
- **Model serialization is `joblib.dump`, not `pickle`** — keep training tasks set to `serializer="pickle"` only at the Celery level (so sklearn objects survive the queue), not for on-disk persistence.
- **JWT tokens are forwarded to Kheops verbatim**: `g.token` set by `routes/utils.py:validate_decorate` is the same token Kheops will see. Never log it.
- **Two Redis instances**: `redis` is for Celery, `redis-socket` is for Socket.IO message queue. They are not interchangeable — emitting to the wrong one silently drops messages.
- **`__init__.py` at the repo root is empty but present** — leave it alone; removing it can break the way `shared/` is picked up in some install paths.

## Frontend contract

- All REST endpoints expect `Authorization: Bearer <jwt>`. Blueprints register `validate_decorate(request)` in `@bp.before_request`.
- Live progress is emitted on Socket.IO events `extraction-status`, `feature-status`, `training-status`.
- CORS origin allowlist is `CORS_ALLOWED_ORIGINS` (comma-separated). The frontend dev server defaults to `http://localhost:3000`.

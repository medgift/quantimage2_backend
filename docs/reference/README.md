# Backend reference (long-form)

Detailed reference material for the QuantImage v2 backend, split by topic so each
file can be read on demand without loading the whole thing into context. The
top-level [CLAUDE.md](../../CLAUDE.md) stays lean and points here.

> Recovered from the former `.github/copilot-instructions.md` and split by topic.
> It is reference detail, not a contract — verify specifics (line numbers, exact
> APIs) against the current code before relying on them.

| File | Covers |
|---|---|
| [architecture.md](architecture.md) | Project overview, data flow, repository structure, tech stack |
| [coding-standards.md](coding-standards.md) | Python style, typing, error handling, logging, imports, formatting (Black), testing |
| [database.md](database.md) | SQLAlchemy model patterns, BaseModel/ORM helpers, connection/pool, migrations |
| [auth.md](auth.md) | Keycloak OIDC, token validation, per-route auth patterns, medical-data handling |
| [kheops.md](kheops.md) | Kheops DICOMweb client and call patterns |
| [celery.md](celery.md) | Task definitions, queues, chord orchestration, Socket.IO progress, celeryconfig |
| [feature-extraction.md](feature-extraction.md) | Extraction flow, config format, feature storage, feature-ID format & prefixes |
| [ml-modeling.md](ml-modeling.md) | Classification/survival algorithms, training flow, pipeline, clinical features |
| [routes-api.md](routes-api.md) | Blueprint/route patterns, response shapes, key endpoints, frontend contract |
| [docker.md](docker.md) | Services, inter-service networking, shared volume |
| [config.md](config.md) | Environment variables, secrets, key constants |

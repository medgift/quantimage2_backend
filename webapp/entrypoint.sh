#!/bin/bash

# Apply any pending DB migrations on startup when DB_AUTOMIGRATE is set (dev/local only).
# NOTE: this intentionally does NOT run `alembic revision --autogenerate`. Generating a
# migration is a deliberate step you run by hand after changing a model (see README /
# CLAUDE.md "Database migrations"); running it on every boot just creates empty revision
# files. Prod does not set DB_AUTOMIGRATE, so it applies migrations manually.
if [ "$DB_AUTOMIGRATE" == "1"  ]; then
  echo "Applying database migrations (alembic upgrade head)"
  alembic upgrade head
fi

python app.py
#!/usr/bin/env bash

# Generate & Run Automigrations if DB_AUTOMIGRATE is set
if [ $DB_AUTOMIGRATE -eq 1  ]; then
  echo "Going to run automigrations"
  alembic revision --autogenerate
  alembic upgrade head
fi

python app.py
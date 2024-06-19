#!/bin/bash

# Generate & Run Automigrations if DB_AUTOMIGRATE is set
if [ "$DB_AUTOMIGRATE" == "1"  ]; then
  echo "creating folder alembic needs to save migrations file"
  mkdir alembic/versions
  echo "Going to run automigrations"
  alembic revision --autogenerate
  alembic upgrade head
fi

python app.py
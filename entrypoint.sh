#!/bin/bash

set -e

echo "Running alembic migrations"
alembic upgrade head

exec "$@"
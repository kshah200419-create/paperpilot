#!/bin/sh
# Container entrypoint: wait for Postgres, migrate, seed the bundled sample once, serve.
set -e

paperpilot db wait --timeout "${DB_WAIT_TIMEOUT:-90}"
paperpilot db upgrade

if [ "${SEED_ON_START:-true}" = "true" ]; then
  paperpilot seed --if-empty
fi

exec uvicorn paperpilot.api.main:app --host 0.0.0.0 --port 8000 --proxy-headers \
  --forwarded-allow-ips="*" --workers "${WEB_CONCURRENCY:-1}"

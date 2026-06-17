#!/bin/sh
# Railway injects $PORT at runtime. This script ensures proper shell expansion.
exec gunicorn app:app \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 1 \
    --threads 2 \
    --timeout 120

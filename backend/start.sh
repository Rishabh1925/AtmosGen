#!/bin/bash
# Render startup script
echo "Starting AtmosGen API..."
exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 120 --graceful-timeout 30
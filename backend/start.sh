#!/bin/bash
# Render startup script
echo "Starting AtmosGen minimal API..."
exec gunicorn main_minimal:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT
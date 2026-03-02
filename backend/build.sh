#!/usr/bin/env bash
# Render build script for AtmosGen

set -o errexit  # exit on error

echo "🔧 Installing Python dependencies..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "✅ Build completed successfully!"
#!/bin/bash

echo "🚀 Starting AtmosGen Backend..."
echo "================================"

cd backend

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🔥 Starting FastAPI server..."
python main.py
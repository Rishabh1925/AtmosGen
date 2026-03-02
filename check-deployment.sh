#!/bin/bash

echo "🔍 AtmosGen Deployment Status Checker"
echo "====================================="

# Check if URLs are provided
if [ $# -eq 0 ]; then
    echo "Usage: ./check-deployment.sh <backend-url> [frontend-url]"
    echo "Example: ./check-deployment.sh https://atmosgen-backend.railway.app https://atmosgen-frontend.vercel.app"
    exit 1
fi

BACKEND_URL=$1
FRONTEND_URL=$2

echo "🔧 Checking Backend: $BACKEND_URL"
echo "-----------------------------------"

# Check health endpoint
echo -n "Health check: "
if curl -s "$BACKEND_URL/health" | grep -q "healthy"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# Check API docs
echo -n "API docs: "
if curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/docs" | grep -q "200"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# Check CORS
echo -n "CORS headers: "
if curl -s -I "$BACKEND_URL/health" | grep -q "access-control-allow-origin"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

if [ ! -z "$FRONTEND_URL" ]; then
    echo ""
    echo "🎨 Checking Frontend: $FRONTEND_URL"
    echo "-----------------------------------"
    
    echo -n "Frontend loading: "
    if curl -s -o /dev/null -w "%{http_code}" "$FRONTEND_URL" | grep -q "200"; then
        echo "✅ PASS"
    else
        echo "❌ FAIL"
    fi
fi

echo ""
echo "🧪 Test these manually:"
echo "- Registration: Try creating an account"
echo "- Login: Test authentication"
echo "- Prediction: Upload images and generate forecast"
echo "- Database: Check if data persists"

echo ""
echo "📊 Useful URLs:"
echo "- Backend Health: $BACKEND_URL/health"
echo "- API Documentation: $BACKEND_URL/docs"
if [ ! -z "$FRONTEND_URL" ]; then
    echo "- Frontend: $FRONTEND_URL"
fi
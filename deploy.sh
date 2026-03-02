#!/bin/bash

echo "🚀 AtmosGen Deployment Helper"
echo "============================="

echo "📋 Pre-deployment checklist:"
echo "✅ Code committed to GitHub"
echo "✅ Environment variables ready"
echo "✅ Database configured"
echo ""

echo "🔗 Quick Deploy Links:"
echo ""
echo "Backend (Railway): https://railway.app/new"
echo "Frontend (Vercel): https://vercel.com/new"
echo ""

echo "📝 Don't forget to set these environment variables:"
echo ""
echo "Backend:"
echo "- PORT=8000"
echo "- ENVIRONMENT=production"
echo "- CORS_ORIGINS=https://your-frontend-url"
echo ""
echo "Frontend:"
echo "- VITE_API_URL=https://your-backend-url"
echo ""

echo "🎯 After deployment, test these URLs:"
echo "- Backend health: https://your-backend-url/health"
echo "- API docs: https://your-backend-url/docs"
echo "- Frontend: https://your-frontend-url"
echo ""

echo "Happy deploying! 🎉"
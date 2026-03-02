# ✅ Render Deployment Checklist

## Pre-Deployment ✅

- [x] **Optimized for deployment** (500MB vs 6.8GB)
- [x] **CPU-only PyTorch** (no CUDA dependencies)
- [x] **Lightweight model** (works without checkpoints)
- [x] **CORS configured** for Render domains
- [x] **Environment variables** ready
- [x] **Build script** created
- [x] **Health checks** configured

## Deploy to Render 🚀

### Step 1: Create Web Service
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click **"New +"** → **"Web Service"**
4. Connect your **AtmosGen** repository

### Step 2: Configure Service (Root Directory = backend)
```
Name: atmosgen-backend
Environment: Python 3
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT
```

### Step 3: Environment Variables
```
USE_LIGHTWEIGHT_MODEL = true
ENVIRONMENT = production
PYTHON_VERSION = 3.11
```

### Step 4: Deploy!
Click **"Create Web Service"** - Render will build and deploy automatically

## Expected Results ✅

- **Build time**: ~3-5 minutes
- **App URL**: `https://atmosgen-backend.onrender.com`
- **Health check**: `https://your-app.onrender.com/health`
- **API docs**: `https://your-app.onrender.com/docs`

## Test Endpoints 🧪

After deployment, verify these work:
- [ ] `GET /health` - Returns 200 OK
- [ ] `GET /docs` - Shows API documentation
- [ ] `POST /register` - User registration
- [ ] `POST /predict` - Weather prediction (upload image)

## Frontend Deployment 🎨

Deploy frontend to **Vercel** or **Netlify**:
- Set `VITE_API_URL = https://your-render-app.onrender.com`
- Root directory: `frontend`

## Free Tier Limits 💰

- **750 hours/month** (24/7 for 31 days)
- **Sleeps after 15 minutes** of inactivity
- **Cold start**: ~30 seconds wake-up time
- **Perfect for demos** and development

## Troubleshooting 🔧

**Build fails?**
- Check Python version is 3.11
- Verify requirements.txt path
- Check build logs in Render dashboard

**App won't start?**
- Verify start command
- Check environment variables
- Review application logs

**CORS errors?**
- Render domains already configured
- Check frontend environment variables

## Ready! 🎯

Your AtmosGen is optimized and ready for Render deployment. The lightweight model will handle weather predictions perfectly for demos and testing.

**Estimated deployment time: 5 minutes** ⏱️
# 🚀 Deploy AtmosGen to Render

## Why Render?

Render is perfect for AtmosGen because:
- ✅ **Free tier** with 750 hours/month
- ✅ **Automatic deployments** from GitHub
- ✅ **Built-in SSL** and custom domains
- ✅ **Easy environment variables**
- ✅ **Great for Python apps**

## Quick Deploy (3 minutes)

### 1. Deploy Backend to Render

1. **Go to [render.com](https://render.com)**
2. **Sign up with GitHub**
3. **Click "New +" → "Web Service"**
4. **Connect your AtmosGen repository**
5. **Configure the service (Root Directory = backend):**

```
Name: atmosgen-backend
Environment: Python 3
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT
```

6. **Add Environment Variables:**
```
USE_LIGHTWEIGHT_MODEL = true
ENVIRONMENT = production
PYTHON_VERSION = 3.11
```

7. **Click "Create Web Service"**

### 2. Deploy Frontend to Vercel/Netlify

**Option A: Vercel**
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repo
3. Set **Root Directory**: `frontend`
4. Add environment variable:
   ```
   VITE_API_URL = https://your-render-app.onrender.com
   ```

**Option B: Netlify**
1. Go to [netlify.com](https://netlify.com)
2. Import your GitHub repo
3. Set **Base directory**: `frontend`
4. Set **Build command**: `npm run build`
5. Set **Publish directory**: `frontend/dist`
6. Add environment variable:
   ```
   VITE_API_URL = https://your-render-app.onrender.com
   ```

## Render Configuration Files

Your project includes optimized Render configuration:

- `render.yaml` - Service configuration
- `backend/build.sh` - Build script
- Optimized `requirements.txt` with CPU-only PyTorch
- `.dockerignore` to exclude large files

## What Works Immediately

✅ **User authentication** (Supabase + SQLite fallback)  
✅ **Weather prediction** (lightweight model)  
✅ **Image upload and processing**  
✅ **User dashboard and history**  
✅ **API documentation** at `/docs`  
✅ **Health checks** at `/health`  

## Testing Your Deployment

After deployment, test these endpoints:
- **Health**: `https://your-app.onrender.com/health`
- **API Docs**: `https://your-app.onrender.com/docs`
- **Upload image** for weather prediction

## Render Free Tier Limits

- **750 hours/month** (enough for 24/7 operation)
- **Sleeps after 15 minutes** of inactivity
- **Cold start time**: ~30 seconds
- **Perfect for development** and demos

## Upgrading to Paid Plan

For production use, consider Render's paid plans:
- **$7/month**: No sleep, faster builds
- **$25/month**: More resources, priority support

## Environment Variables for Production

If using Supabase (recommended for production):

```
USE_LIGHTWEIGHT_MODEL = true
ENVIRONMENT = production
SUPABASE_URL = your_supabase_url
SUPABASE_ANON_KEY = your_supabase_anon_key
SUPABASE_SERVICE_KEY = your_supabase_service_key
```

## Troubleshooting

**Build fails?**
- Check build logs in Render dashboard
- Verify `requirements.txt` is correct
- Ensure Python version is 3.11

**App won't start?**
- Check start command is correct
- Verify environment variables are set
- Check application logs

**CORS errors?**
- Add your frontend URL to CORS origins
- Check environment variables

## Ready to Deploy! 🎯

1. **Your code is already optimized** for Render
2. **Follow the steps above** 
3. **Your app will be live** in ~5 minutes

Render will automatically redeploy when you push to GitHub! 🚀
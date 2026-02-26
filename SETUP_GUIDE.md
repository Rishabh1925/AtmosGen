# 🚀 AtmosGen Setup Guide

## Phase 1: Local Production Prototype

This guide will help you set up and run the AtmosGen full-stack application locally.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.8+** installed
- **Node.js 16+** installed  
- **npm** or **yarn** package manager
- **Git** (for cloning)

---

## 🛠️ Installation Steps

### Step 1: Clone and Navigate
```bash
git clone <your-repo-url>
cd AtmosGen
```

### Step 2: Backend Setup

**Option A: Using the startup script (Recommended)**
```bash
./start_backend.sh
```

**Option B: Manual setup**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend will start on `http://localhost:8000`

### Step 3: Frontend Setup (New Terminal)

**Option A: Using the startup script (Recommended)**
```bash
./start_frontend.sh
```

**Option B: Manual setup**
```bash
cd frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:3000`

---

## 🧪 Testing the Setup

### Test 1: API Health Check
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Test 2: Run Test Suite
```bash
python test_api.py
```

### Test 3: Web Interface
1. Open browser to `http://localhost:3000`
2. You should see the AtmosGen interface
3. Status should show "✅ Ready"

---

## 📸 Using the Application

### Step 1: Prepare Images
- Use satellite images (PNG/JPG format)
- Recommended: 3-8 images in sequence
- Images will be automatically resized to 128x128

### Step 2: Upload Sequence
1. Click "📁 Select Images (Max 10)"
2. Choose your satellite image sequence
3. Preview will show uploaded images

### Step 3: Generate Forecast
1. Click "🚀 Generate Forecast"
2. Wait for processing (loading indicator)
3. View generated future frame

---

## 🔧 API Usage Examples

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Generate Forecast
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "files=@image3.png"
```

### Python Example
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
files = [
    ('files', ('img1.png', open('img1.png', 'rb'), 'image/png')),
    ('files', ('img2.png', open('img2.png', 'rb'), 'image/png')),
]
response = requests.post("http://localhost:8000/predict", files=files)
result = response.json()
print(f"Generated image: {len(result['generated_image'])} chars")
```

---

## 🐛 Troubleshooting

### Backend Issues

**"Model not loaded" error:**
- Check if checkpoint files exist in `checkpoints/` directory
- Backend will use random weights if no checkpoint found
- This is normal for first run

**Port 8000 already in use:**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9
# Or change port in backend/main.py
```

**Import errors:**
```bash
# Ensure you're in the right directory
cd backend
# Check Python path
python -c "import sys; print(sys.path)"
```

### Frontend Issues

**Port 3000 already in use:**
- Vite will automatically suggest port 3001
- Or kill existing process: `lsof -ti:3000 | xargs kill -9`

**CORS errors:**
- Ensure backend is running on port 8000
- Check CORS settings in `backend/main.py`

**Build errors:**
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### General Issues

**"Connection refused" errors:**
- Ensure both backend and frontend are running
- Check firewall settings
- Verify ports 8000 and 3000 are accessible

---

## 📊 Performance Notes

- **First prediction:** May take 10-30 seconds (model loading)
- **Subsequent predictions:** 2-5 seconds typically
- **Memory usage:** ~2-4GB RAM (depends on model size)
- **GPU:** Automatically detected and used if available

---

## 🔄 Development Workflow

### Backend Development
```bash
cd backend
# Auto-reload on changes
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
# Hot reload enabled by default
npm run dev
```

### Model Training
```bash
# Train new model (optional)
python main.py
```

---

## 📁 File Structure Reference

```
AtmosGen/
├── backend/           # FastAPI backend
├── frontend/          # React frontend  
├── core_model/        # ML model code
├── checkpoints/       # Model weights
├── start_backend.sh   # Backend startup script
├── start_frontend.sh  # Frontend startup script
├── test_api.py        # API test suite
└── README.md          # Main documentation
```

---

## 🎯 Next Steps

Once you have the basic system running:

1. **Train your own model:** Use `python main.py` with your data
2. **Customize the UI:** Modify `frontend/src/App.jsx`
3. **Add features:** Extend the API in `backend/main.py`
4. **Deploy:** Prepare for Phase 2 (authentication, database)

---

## 💡 Tips

- **Use sample images:** Test with the provided satellite images first
- **Monitor logs:** Check console output for debugging
- **Browser dev tools:** Use F12 to debug frontend issues
- **API testing:** Use tools like Postman or curl for API testing

---

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review console logs (backend and frontend)
3. Test API endpoints individually
4. Verify all dependencies are installed

---

**Happy forecasting! 🌤️**
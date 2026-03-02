"""
Minimal AtmosGen API for quick Render deployment
Stripped down version without heavy dependencies
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv
import base64
import io
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AtmosGen API - Minimal",
    description="Minimal version for deployment testing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173",
        "https://*.vercel.app",
        "https://*.netlify.app", 
        "https://*.railway.app",
        "https://*.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AtmosGen API is running (minimal version)",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AtmosGen API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/predict")
async def predict_minimal():
    """Minimal prediction endpoint - returns demo response"""
    
    # Create a simple demo image (blue gradient)
    width, height = 256, 256
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a blue gradient pattern
    for y in range(height):
        for x in range(width):
            image_array[y, x] = [
                int(50 + (x / width) * 100),  # Red
                int(100 + (y / height) * 100),  # Green  
                int(150 + ((x + y) / (width + height)) * 100)  # Blue
            ]
    
    # Convert to PIL Image
    image_pil = Image.fromarray(image_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return {
        "generated_image": image_b64,
        "processing_time": 0.1,
        "model_type": "Demo Pattern Generator",
        "message": "This is a minimal demo - full model will be added after deployment",
        "status": "success"
    }

@app.get("/api/satellite/regions")
async def get_satellite_regions():
    """Demo satellite regions"""
    return {
        "regions": [
            {"id": "india", "name": "India", "bounds": [68, 8, 97, 37]},
            {"id": "asia", "name": "Asia", "bounds": [60, 5, 150, 50]}
        ]
    }

@app.get("/api/satellite/layers")
async def get_satellite_layers():
    """Demo satellite layers"""
    return {
        "layers": [
            {"id": "visible", "name": "Visible", "description": "Visible light imagery"},
            {"id": "infrared", "name": "Infrared", "description": "Thermal infrared imagery"}
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "status": "error"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "status": "error"}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
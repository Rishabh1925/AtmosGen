from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import List
import asyncio
from contextlib import asynccontextmanager

from model_service import ModelService
from schemas import PredictionResponse, HealthResponse
from utils import setup_logging

# Global model service instance
model_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup"""
    global model_service
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AtmosGen backend...")
    
    try:
        # Initialize model service
        model_service = ModelService()
        await model_service.load_model()
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    finally:
        logger.info("Shutting down AtmosGen backend...")

# Create FastAPI app
app = FastAPI(
    title="AtmosGen API",
    description="Physics-Informed Diffusion Model for Satellite Weather Nowcasting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service is not None and model_service.is_loaded()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(files: List[UploadFile] = File(...)):
    """
    Generate weather forecast from satellite image sequence
    
    Args:
        files: List of satellite images (sequence input)
        
    Returns:
        Generated future frame as base64 encoded image
    """
    global model_service
    
    if not model_service or not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(files) > 10:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Too many images (max 10)")
    
    try:
        logger.info(f"Processing prediction request with {len(files)} images")
        
        # Process images and generate prediction
        result = await model_service.predict(files)
        
        logger.info("Prediction completed successfully")
        return PredictionResponse(
            success=True,
            generated_image=result["generated_image"],
            processing_time=result["processing_time"],
            message="Forecast generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
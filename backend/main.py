from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn
import logging
from typing import List, Optional, Annotated
import asyncio
from contextlib import asynccontextmanager

from model_service import ModelService
from schemas import PredictionResponse, HealthResponse, UserRegister, UserLogin, UserResponse, ForecastResponse, ForecastListResponse
from utils import setup_logging
from database import Database

# Global instances
model_service = None
db = None
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model and init database on startup"""
    global model_service, db
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AtmosGen backend...")
    
    try:
        # Initialize database
        db = Database()
        logger.info("Database initialized successfully")
        
        # Initialize model service
        model_service = ModelService()
        await model_service.load_model()
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        logger.info("Shutting down AtmosGen backend...")

# Create FastAPI app
app = FastAPI(
    title="AtmosGen API",
    description="Physics-Informed Diffusion Model for Satellite Weather Nowcasting",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173",
        "https://*.vercel.app",  # Allow all Vercel deployments
        "https://*.netlify.app", # Allow all Netlify deployments
        "https://*.railway.app", # Allow Railway deployments
        "https://*.onrender.com" # Allow Render deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Dependency to get current user from session
async def get_current_user(session_token: Annotated[str | None, Cookie()] = None):
    """Get current user from session token"""
    if not session_token:
        return None
    
    user = db.get_user_by_session(session_token)
    return user

# Dependency that requires authentication
async def require_auth(current_user = Depends(get_current_user)):
    """Require user to be authenticated"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service is not None and model_service.is_loaded()
    )

@app.post("/auth/register", response_model=UserResponse)
async def register(user_data: UserRegister):
    """Register a new user"""
    # Basic validation
    if len(user_data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    if "@" not in user_data.email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    # Create user
    user_id = db.create_user(user_data.username, user_data.email, user_data.password)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    # Create session
    session_token = db.create_session(user_id)
    
    response = JSONResponse(content={
        "success": True,
        "user": {
            "id": user_id,
            "username": user_data.username,
            "email": user_data.email
        },
        "message": "Registration successful"
    })
    
    # Set session cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        max_age=7 * 24 * 60 * 60,  # 7 days
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax"
    )
    
    return response

@app.post("/auth/login", response_model=UserResponse)
async def login(user_data: UserLogin):
    """Login user"""
    user = db.authenticate_user(user_data.username, user_data.password)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create session
    session_token = db.create_session(user['id'])
    
    response = JSONResponse(content={
        "success": True,
        "user": user,
        "message": "Login successful"
    })
    
    # Set session cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        max_age=7 * 24 * 60 * 60,  # 7 days
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax"
    )
    
    return response

@app.post("/auth/logout")
async def logout(session_token: Annotated[str | None, Cookie()] = None):
    """Logout user"""
    if session_token:
        db.logout_session(session_token)
    
    response = JSONResponse(content={"success": True, "message": "Logged out successfully"})
    response.delete_cookie("session_token")
    return response

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user info"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return UserResponse(
        success=True,
        user=current_user,
        message="User info retrieved"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    files: List[UploadFile] = File(...),
    forecast_name: str = "Untitled Forecast",
    current_user = Depends(require_auth)
):
    """
    Generate weather forecast from satellite image sequence
    Requires authentication
    """
    global model_service
    
    if not model_service or not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(files) > 10:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Too many images (max 10)")
    
    try:
        logger.info(f"Processing prediction request from user {current_user['id']} with {len(files)} images")
        
        # Process images and generate prediction
        result = await model_service.predict(files)
        
        # Save forecast to database
        forecast_id = db.save_forecast(
            user_id=current_user['id'],
            name=forecast_name,
            input_images_count=len(files),
            generated_image=result["generated_image"],
            processing_time=result["processing_time"]
        )
        
        logger.info("Prediction completed successfully")
        return PredictionResponse(
            success=True,
            generated_image=result["generated_image"],
            processing_time=result["processing_time"],
            message="Forecast generated successfully",
            forecast_id=forecast_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/forecasts", response_model=ForecastListResponse)
async def get_forecasts(current_user = Depends(require_auth)):
    """Get user's forecast history"""
    try:
        forecasts = db.get_user_forecasts(current_user['id'])
        
        return ForecastListResponse(
            success=True,
            forecasts=forecasts,
            message="Forecasts retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Get forecasts failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve forecasts")

@app.get("/forecasts/{forecast_id}", response_model=ForecastResponse)
async def get_forecast(forecast_id: int, current_user = Depends(require_auth)):
    """Get specific forecast by ID"""
    try:
        forecast = db.get_forecast_by_id(forecast_id, current_user['id'])
        
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
        
        return ForecastResponse(
            success=True,
            forecast=forecast,
            message="Forecast retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get forecast failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve forecast")

if __name__ == "__main__":
    import os
    
    # Get port from environment (for Railway/Render/Heroku)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
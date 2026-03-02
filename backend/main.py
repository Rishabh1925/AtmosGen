from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Cookie, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn
import logging
from typing import List, Optional, Annotated
import asyncio
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from model_service import ModelService
from lightweight_model import LightweightModelService
from schemas import (
    PredictionResponse, HealthResponse, UserRegister, UserLogin, UserResponse, 
    ForecastResponse, ForecastListResponse, UserProfileUpdate, PasswordChange,
    UserProfileResponse, DashboardResponse, ActivityResponse, ForecastUpdate
)
from utils import setup_logging
from supabase_client import SupabaseDB
from sqlite_db import SQLiteDB
from auth_service import supabase_auth, get_current_user, require_auth
from satellite_service import SatelliteService

# Global instances
model_service = None
db = None
satellite_service = None
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model and init database on startup"""
    global model_service, db, satellite_service
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AtmosGen backend...")
    
    try:
        # Initialize database (Supabase or SQLite fallback)
        supabase_db = SupabaseDB()
        if supabase_db.is_connected():
            db = supabase_db
            logger.info("Supabase database connected successfully")
        else:
            db = SQLiteDB()
            logger.info("Using SQLite database (local fallback)")
        
        # Initialize satellite service
        satellite_service = SatelliteService()
        logger.info("Satellite service initialized successfully")
        
        # Initialize model service (use lightweight for production deployment)
        use_lightweight = os.getenv('USE_LIGHTWEIGHT_MODEL', 'false').lower() == 'true'
        if use_lightweight:
            model_service = LightweightModelService()
            logger.info("Using lightweight model for deployment")
        else:
            model_service = ModelService()
            logger.info("Using full model with checkpoints")
        
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

# Unified authentication dependency
async def get_current_user_unified(
    authorization: Annotated[str | None, Header()] = None,
    auth_token: Annotated[str | None, Cookie()] = None
) -> Optional[dict]:
    """Get current user from either Authorization header or auth_token cookie"""
    token = None
    
    # Try Authorization header first (Bearer token)
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    # Fall back to cookie
    elif auth_token:
        token = auth_token
    
    if not token:
        return None
    
    # Try Supabase first, then SQLite
    if isinstance(db, SupabaseDB) and db.is_connected():
        user = await supabase_auth.get_user_from_token(token)
        return user
    else:
        # SQLite session lookup
        user = db.get_user_by_session(token)
        return user

async def require_auth_unified(current_user = Depends(get_current_user_unified)):
    """Require user to be authenticated (unified approach)"""
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
    
    try:
        if isinstance(db, SupabaseDB) and db.is_connected():
            # Use Supabase Auth
            auth_result = await supabase_auth.register_user(
                user_data.email,
                user_data.password,
                {
                    "username": user_data.username,
                    "first_name": user_data.first_name,
                    "last_name": user_data.last_name
                }
            )
            
            # Create user profile in our database
            await db.create_user_profile(auth_result["user"]["id"], {
                "username": user_data.username,
                "first_name": user_data.first_name,
                "last_name": user_data.last_name
            })
            
            response_data = {
                "success": True,
                "user": auth_result["user"],
                "message": "Registration successful. Please check your email to confirm your account."
            }
            
            response = JSONResponse(content=response_data)
            
            # Set auth token as cookie if available
            if auth_result.get("session"):
                response.set_cookie(
                    key="auth_token",
                    value=auth_result["session"],
                    max_age=7 * 24 * 60 * 60,  # 7 days
                    httponly=True,
                    secure=False,  # Set to True in production with HTTPS
                    samesite="lax"
                )
            
            return response
        
        else:
            # Use SQLite fallback
            user_id = db.create_user(
                email=user_data.email,
                password=user_data.password,
                username=user_data.username,
                first_name=user_data.first_name,
                last_name=user_data.last_name
            )
            
            if not user_id:
                raise HTTPException(status_code=400, detail="Email or username already exists")
            
            # Create session
            session_token = db.create_session(user_id)
            
            user_info = {
                "id": user_id,
                "email": user_data.email,
                "username": user_data.username,
                "email_confirmed": True
            }
            
            response_data = {
                "success": True,
                "user": user_info,
                "message": "Registration successful!"
            }
            
            response = JSONResponse(content=response_data)
            
            # Set session cookie
            response.set_cookie(
                key="auth_token",
                value=session_token,
                max_age=7 * 24 * 60 * 60,  # 7 days
                httponly=True,
                secure=False,
                samesite="lax"
            )
            
            return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/auth/login", response_model=UserResponse)
async def login(user_data: UserLogin):
    """Login user"""
    try:
        if isinstance(db, SupabaseDB) and db.is_connected():
            # Use Supabase Auth
            auth_result = await supabase_auth.login_user(user_data.username, user_data.password)
            
            response_data = {
                "success": True,
                "user": auth_result["user"],
                "message": "Login successful"
            }
            
            response = JSONResponse(content=response_data)
            
            # Set auth token as cookie
            response.set_cookie(
                key="auth_token",
                value=auth_result["session"],
                max_age=7 * 24 * 60 * 60,  # 7 days
                httponly=True,
                secure=False,  # Set to True in production with HTTPS
                samesite="lax"
            )
            
            return response
        
        else:
            # Use SQLite fallback
            user = db.authenticate_user(user_data.username, user_data.password)
            
            if not user:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Create session
            session_token = db.create_session(user['id'])
            
            response_data = {
                "success": True,
                "user": user,
                "message": "Login successful"
            }
            
            response = JSONResponse(content=response_data)
            
            # Set session cookie
            response.set_cookie(
                key="auth_token",
                value=session_token,
                max_age=7 * 24 * 60 * 60,  # 7 days
                httponly=True,
                secure=False,
                samesite="lax"
            )
            
            return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/auth/logout")
async def logout(auth_token: Annotated[str | None, Cookie()] = None):
    """Logout user"""
    try:
        if auth_token:
            if isinstance(db, SupabaseDB) and db.is_connected():
                await supabase_auth.logout_user(auth_token)
            else:
                # SQLite session cleanup
                db.delete_session(auth_token)
        
        response = JSONResponse(content={"success": True, "message": "Logged out successfully"})
        response.delete_cookie("auth_token")
        return response
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        response = JSONResponse(content={"success": True, "message": "Logged out successfully"})
        response.delete_cookie("auth_token")
        return response

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user_unified)):
    """Get current user info"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return UserResponse(
        success=True,
        user=current_user,
        message="User info retrieved"
    )

@app.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user = Depends(require_auth_unified)):
    """Get complete user profile with stats"""
    try:
        if db and db.is_connected():
            profile = await db.get_user_profile(current_user['id'])
        else:
            # Fallback to basic user info
            profile = current_user
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return UserProfileResponse(
            success=True,
            profile=profile,
            message="Profile retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")

@app.put("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    profile_data: UserProfileUpdate,
    current_user = Depends(require_auth_unified)
):
    """Update user profile"""
    try:
        if db and db.is_connected():
            success = await db.update_user_profile(current_user['id'], profile_data.dict(exclude_unset=True))
            
            if not success:
                raise HTTPException(status_code=400, detail="Failed to update profile")
            
            # Get updated profile
            updated_profile = await db.get_user_profile(current_user['id'])
        else:
            updated_profile = current_user
        
        return UserProfileResponse(
            success=True,
            profile=updated_profile,
            message="Profile updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update profile failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@app.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_stats(current_user = Depends(require_auth_unified)):
    """Get user dashboard statistics"""
    try:
        if db and db.is_connected():
            stats = await db.get_dashboard_stats(current_user['id'])
        else:
            stats = {
                "total_forecasts": 0,
                "today_forecasts": 0,
                "week_forecasts": 0,
                "avg_processing_time": 0.0,
                "recent_forecasts": []
            }
        
        return DashboardResponse(
            success=True,
            stats=stats,
            message="Dashboard stats retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Get dashboard stats failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard stats")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    files: List[UploadFile] = File(...),
    forecast_name: str = "Untitled Forecast",
    current_user = Depends(require_auth_unified)
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
        forecast_id = None
        if isinstance(db, SupabaseDB) and db.is_connected():
            # Update user stats
            await db.update_user_stats(current_user['id'], result["processing_time"])
            
            # Save forecast
            forecast_id = await db.save_forecast(current_user['id'], {
                "name": forecast_name,
                "input_images_count": len(files),
                "generated_image": result["generated_image"],
                "processing_time": result["processing_time"]
            })
        else:
            # SQLite fallback
            forecast_id = db.save_forecast(current_user['id'], {
                "name": forecast_name,
                "input_images_count": len(files),
                "generated_image": result["generated_image"],
                "processing_time": result["processing_time"]
            })
        
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
async def get_forecasts(current_user = Depends(require_auth_unified)):
    """Get user's forecast history"""
    try:
        if isinstance(db, SupabaseDB) and db.is_connected():
            forecasts = await db.get_user_forecasts(current_user['id'])
        else:
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
async def get_forecast(forecast_id: int, current_user = Depends(require_auth_unified)):
    """Get specific forecast by ID"""
    try:
        if isinstance(db, SupabaseDB) and db.is_connected():
            forecast = await db.get_forecast_by_id(str(forecast_id), current_user['id'])
        else:
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

# Satellite Data Endpoints

@app.get("/satellite/regions")
async def get_satellite_regions():
    """Get available satellite regions"""
    try:
        regions = await satellite_service.get_regions_list()
        return {"success": True, "regions": regions}
    except Exception as e:
        logger.error(f"Get regions failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get regions")

@app.get("/satellite/layers")
async def get_satellite_layers():
    """Get available satellite layers"""
    try:
        layers = await satellite_service.get_layers_list()
        return {"success": True, "layers": layers}
    except Exception as e:
        logger.error(f"Get layers failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get layers")

@app.get("/satellite/dates")
async def get_available_dates(days_back: int = 7):
    """Get available dates for satellite data"""
    try:
        dates = await satellite_service.get_available_dates(days_back)
        return {"success": True, "dates": dates}
    except Exception as e:
        logger.error(f"Get dates failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dates")

@app.get("/satellite/sequence")
async def get_satellite_sequence(
    region: str,
    layer: str = "visible",
    sequence_length: int = 4,
    end_date: Optional[str] = None,
    current_user = Depends(require_auth_unified)
):
    """Get satellite image sequence for ML prediction"""
    try:
        logger.info(f"Fetching satellite sequence for user {current_user['id']}: region={region}, layer={layer}")
        
        images = await satellite_service.get_satellite_sequence(
            region=region,
            layer=layer,
            sequence_length=sequence_length,
            end_date=end_date
        )
        
        if not images:
            raise HTTPException(status_code=404, detail="No satellite images found for the specified parameters")
        
        return {
            "success": True,
            "images": images,
            "region": region,
            "layer": layer,
            "sequence_length": len(images),
            "message": f"Retrieved {len(images)} satellite images"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get satellite sequence failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve satellite images")

@app.post("/predict/satellite")
async def predict_from_satellite(
    region: str,
    layer: str = "visible",
    sequence_length: int = 4,
    end_date: Optional[str] = None,
    forecast_name: str = "Satellite Forecast",
    current_user = Depends(require_auth_unified)
):
    """
    Generate weather forecast from real satellite data
    Automatically fetches satellite images and runs prediction
    """
    global model_service, satellite_service
    
    if not model_service or not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Processing satellite prediction for user {current_user['id']}: region={region}")
        
        # Fetch satellite image sequence
        images = await satellite_service.get_satellite_sequence(
            region=region,
            layer=layer,
            sequence_length=sequence_length,
            end_date=end_date
        )
        
        if not images:
            raise HTTPException(status_code=404, detail="No satellite images available")
        
        # Convert images to format expected by model
        # For now, we'll use the base64 images directly
        # In production, you'd convert them to proper format
        
        # Create mock UploadFile objects from the satellite images
        from fastapi import UploadFile
        import io
        
        upload_files = []
        for img_data in images:
            # Extract base64 data
            base64_data = img_data["image_data"].split(",")[1]
            image_bytes = base64.b64decode(base64_data)
            
            # Create UploadFile-like object
            file_obj = io.BytesIO(image_bytes)
            upload_file = UploadFile(
                filename=f"satellite_{img_data['date']}.jpg",
                file=file_obj,
                content_type="image/jpeg"
            )
            upload_files.append(upload_file)
        
        # Run prediction
        result = await model_service.predict(upload_files)
        
        # Save forecast to database with satellite metadata
        forecast_name_full = f"{forecast_name} ({region}, {layer})"
        forecast_id = None
        if db and db.is_connected():
            forecast_id = await db.save_forecast(
                user_id=current_user['id'],
                forecast_data={
                    "name": forecast_name_full,
                    "input_images_count": len(images),
                    "generated_image": result["generated_image"],
                    "processing_time": result["processing_time"],
                    "region": region,
                    "layer": layer,
                    "source_type": "satellite"
                }
            )
        
        logger.info("Satellite prediction completed successfully")
        return PredictionResponse(
            success=True,
            generated_image=result["generated_image"],
            processing_time=result["processing_time"],
            message=f"Satellite forecast generated for {region}",
            forecast_id=forecast_id,
            metadata={
                "region": region,
                "layer": layer,
                "sequence_length": len(images),
                "source": "Real Satellite Data"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Satellite prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Satellite prediction failed: {str(e)}")

@app.get("/satellite/health")
async def satellite_health_check():
    """Check satellite data services health"""
    try:
        health = await satellite_service.health_check()
        return {"success": True, "services": health}
    except Exception as e:
        logger.error(f"Satellite health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

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
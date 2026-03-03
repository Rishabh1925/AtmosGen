"""
AtmosGen API — Cloud Coverage Prediction from Satellite Imagery

Uses a fine-tuned U-Net (EfficientNet-B0 encoder) for cloud segmentation
on GOES-18 Band 13 infrared satellite data.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Cookie, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn
import logging
from typing import List, Optional, Annotated
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Import MongoDB client
from mongodb_client import mongodb

# Import cloud segmentation model
from cloud_model import CloudModelService

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    generated_image: str
    cloud_mask: str
    input_image: str
    cloud_coverage_pct: float
    processing_time: float
    model_type: str
    message: str
    status: str


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    model_loaded: bool
    environment: str


class UserRegister(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str


security = HTTPBearer(auto_error=False)

# Global model service
cloud_model = CloudModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger = logging.getLogger(__name__)
    logger.info("Starting AtmosGen backend...")

    try:
        # Connect to MongoDB
        connected = await mongodb.connect()
        if connected:
            logger.info("MongoDB connected successfully")
        else:
            logger.warning("MongoDB connection failed — auth/history features disabled")

        # Model loads lazily on first prediction request (avoids Gunicorn worker timeout)
        logger.info("Cloud model will load on first prediction request (lazy loading)")

        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        logger.info("Shutting down AtmosGen backend...")
        await mongodb.close()


# Create FastAPI app
app = FastAPI(
    title="AtmosGen API",
    description="AI-Powered Cloud Coverage Prediction from Satellite Imagery",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://atmos-gen.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.(vercel\.app|netlify\.app|railway\.app|onrender\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Authentication ───────────────────────────────────────────

async def get_current_user_unified(
    authorization: Annotated[str | None, Header()] = None,
    auth_token: Annotated[str | None, Cookie()] = None
) -> Optional[dict]:
    """Get current user from either Authorization header or auth_token cookie"""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    elif auth_token:
        token = auth_token

    if not token:
        return None

    user = await mongodb.get_user_by_token(token)
    return user


async def require_auth_unified(current_user=Depends(get_current_user_unified)):
    """Require authentication"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


# ─── Health & Info ────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="AtmosGen API is running",
        version="3.0.0",
        model_loaded=cloud_model.is_loaded(),
        environment=os.getenv("ENVIRONMENT", "development")
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AtmosGen API",
        "status": "running",
        "model": "Cloud Segmentation (U-Net + EfficientNet-B0)",
        "model_loaded": cloud_model.is_loaded(),
        "docs": "/docs",
        "health": "/health"
    }


# ─── Auth Endpoints ──────────────────────────────────────────

@app.post("/register")
async def register(user_data: UserRegister):
    """Register a new user"""
    user = await mongodb.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password
    )

    if not user:
        raise HTTPException(status_code=400, detail="User already exists")

    token = await mongodb.create_session(user["id"])

    response = JSONResponse(content={
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"]
        },
        "message": "User registered successfully"
    })

    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="none"
    )

    return response


@app.post("/login")
async def login(user_data: UserLogin):
    """Login user"""
    user = await mongodb.get_user_by_credentials(
        username=user_data.username,
        password=user_data.password
    )

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = await mongodb.create_session(user["id"])

    response = JSONResponse(content={
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"]
        },
        "message": "Login successful"
    })

    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="none"
    )

    return response


@app.post("/logout")
async def logout(auth_token: Annotated[str | None, Cookie()] = None):
    """Logout user"""
    if auth_token:
        await mongodb.delete_session(auth_token)

    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("auth_token")
    return response


@app.get("/me")
async def get_current_user_info(current_user=Depends(get_current_user_unified)):
    """Get current user info"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {"user": current_user}


# ─── Prediction ──────────────────────────────────────────────

@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    current_user=Depends(get_current_user_unified)
):
    """
    Generate cloud coverage prediction from uploaded satellite image.

    Returns cloud segmentation mask, coverage percentage, and visualization.
    """
    # Lazy-load the model on first request
    if not cloud_model.is_loaded():
        logger.info("First prediction request — loading cloud model now...")
        loaded = await cloud_model.load_model()
        if not loaded:
            raise HTTPException(
                status_code=503,
                detail="Model failed to load. Please ensure checkpoint exists."
            )

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        # Run cloud segmentation
        result = await cloud_model.predict(files)

        # Save forecast to MongoDB if user is authenticated
        if current_user:
            forecast_data = {
                "location": "Satellite",
                "processing_time": result["processing_time"],
                "model_type": result["model_type"],
                "cloud_coverage_pct": result["cloud_coverage_pct"],
                "file_name": files[0].filename
            }
            await mongodb.create_forecast(current_user["id"], forecast_data)

        return {
            "generated_image": result["generated_image"],
            "cloud_mask": result["cloud_mask"],
            "input_image": result["input_image"],
            "cloud_coverage_pct": result["cloud_coverage_pct"],
            "processing_time": result["processing_time"],
            "model_type": result["model_type"],
            "message": "Cloud coverage prediction generated successfully",
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─── Dashboard & History ─────────────────────────────────────

@app.get("/dashboard")
async def get_dashboard_stats(current_user=Depends(require_auth_unified)):
    """Get dashboard statistics — real metrics from forecast history"""
    stats = await mongodb.get_dashboard_stats(current_user["id"])
    return stats


@app.get("/forecasts")
async def get_forecasts(current_user=Depends(require_auth_unified)):
    """Get user's forecast history"""
    forecasts = await mongodb.get_user_forecasts(current_user["id"])

    formatted_forecasts = []
    for forecast in forecasts:
        formatted_forecasts.append({
            "id": forecast["id"],
            "location": forecast.get("location", "Satellite"),
            "created_at": forecast["created_at"].isoformat(),
            "cloud_coverage_pct": forecast.get("cloud_coverage_pct", 0),
            "model_type": forecast.get("model_type", "Unknown"),
            "status": "completed"
        })

    return {"forecasts": formatted_forecasts}


@app.get("/forecasts/{forecast_id}")
async def get_forecast(forecast_id: str, current_user=Depends(require_auth_unified)):
    """Get specific forecast"""
    forecast = await mongodb.get_forecast_by_id(forecast_id, current_user["id"])

    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not found")

    return {"forecast": forecast}


# ─── Satellite Info ──────────────────────────────────────────

@app.get("/api/satellite/regions")
async def get_satellite_regions():
    """Get available satellite regions"""
    return {
        "regions": [
            {"id": "conus", "name": "CONUS (Continental US)", "bounds": [-130, 20, -60, 55]},
            {"id": "full_disk", "name": "Full Disk", "bounds": [-156, -81, 6, 81]},
        ]
    }


@app.get("/api/satellite/layers")
async def get_satellite_layers():
    """Get available satellite layers"""
    return {
        "layers": [
            {"id": "band13", "name": "Band 13 (10.3μm IR)", "description": "Clean longwave IR window — used for cloud detection"},
            {"id": "band02", "name": "Band 2 (Visible)", "description": "Red visible band"},
        ]
    }


# ─── Error Handlers ──────────────────────────────────────────

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
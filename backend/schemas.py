from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    success: bool
    generated_image: str  # Base64 encoded image
    processing_time: float
    message: str
    forecast_id: Optional[int] = None

# Authentication schemas
class UserRegister(BaseModel):
    """User registration schema"""
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    """User login schema"""
    username: str  # Can be username or email
    password: str

class UserResponse(BaseModel):
    """User response schema"""
    success: bool
    user: Optional[Dict[str, Any]] = None
    message: str

# Forecast schemas
class ForecastItem(BaseModel):
    """Individual forecast item"""
    id: int
    name: Optional[str]
    input_images_count: int
    processing_time: float
    created_at: str

class ForecastDetail(BaseModel):
    """Detailed forecast with image"""
    id: int
    name: Optional[str]
    input_images_count: int
    generated_image: str
    processing_time: float
    created_at: str

class ForecastListResponse(BaseModel):
    """Forecast list response"""
    success: bool
    forecasts: List[ForecastItem]
    message: str

class ForecastResponse(BaseModel):
    """Single forecast response"""
    success: bool
    forecast: ForecastDetail
    message: str
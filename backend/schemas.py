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
    metadata: Optional[Dict[str, Any]] = None

# Authentication schemas
class UserRegister(BaseModel):
    """User registration schema"""
    username: str
    email: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    """User login schema"""
    username: str  # Can be username or email
    password: str

class UserResponse(BaseModel):
    """User response schema"""
    success: bool
    user: Optional[Dict[str, Any]] = None
    message: str

class UserProfileUpdate(BaseModel):
    """User profile update schema"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    email_notifications: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None

class PasswordChange(BaseModel):
    """Password change schema"""
    old_password: str
    new_password: str

class UserProfileResponse(BaseModel):
    """User profile response schema"""
    success: bool
    profile: Optional[Dict[str, Any]] = None
    message: str

class DashboardResponse(BaseModel):
    """Dashboard stats response schema"""
    success: bool
    stats: Optional[Dict[str, Any]] = None
    message: str

class ActivityResponse(BaseModel):
    """User activity response schema"""
    success: bool
    activities: Optional[List[Dict[str, Any]]] = None
    message: str

# Forecast schemas
class ForecastItem(BaseModel):
    """Individual forecast item"""
    id: int
    name: Optional[str]
    description: Optional[str] = None
    input_images_count: int
    processing_time: float
    model_version: Optional[str] = None
    region: Optional[str] = None
    layer: Optional[str] = None
    source_type: Optional[str] = None
    is_favorite: Optional[bool] = False
    view_count: Optional[int] = 0
    created_at: str

class ForecastDetail(BaseModel):
    """Detailed forecast with image"""
    id: int
    name: Optional[str]
    description: Optional[str] = None
    input_images_count: int
    generated_image: str
    processing_time: float
    model_version: Optional[str] = None
    region: Optional[str] = None
    layer: Optional[str] = None
    source_type: Optional[str] = None
    is_favorite: Optional[bool] = False
    is_public: Optional[bool] = False
    view_count: Optional[int] = 0
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

class ForecastUpdate(BaseModel):
    """Forecast update schema"""
    name: Optional[str] = None
    description: Optional[str] = None
    is_favorite: Optional[bool] = None
    is_public: Optional[bool] = None
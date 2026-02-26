from pydantic import BaseModel
from typing import Optional

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
    input_sequence_length: Optional[int] = None
"""
MongoDB client for AtmosGen
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime
import logging
from typing import Optional, Dict, List
import hashlib

logger = logging.getLogger(__name__)

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None
        self.users = None
        self.forecasts = None
        self.sessions = None
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            import certifi
            
            # MongoDB connection string from environment
            mongodb_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
            db_name = os.getenv('MONGODB_DB', 'atmosgen')
            
            # Use certifi CA bundle for SSL (required on Render/cloud platforms)
            self.client = AsyncIOMotorClient(mongodb_url, tlsCAFile=certifi.where())
            self.db = self.client[db_name]
            
            # Collections
            self.users = self.db.users
            self.forecasts = self.db.forecasts
            self.sessions = self.db.sessions
            
            # Create indexes
            await self._create_indexes()
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {db_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    async def _create_indexes(self):
        """Create database indexes"""
        try:
            # User indexes
            await self.users.create_index("username", unique=True)
            await self.users.create_index("email", unique=True)
            
            # Session indexes
            await self.sessions.create_index("token", unique=True)
            await self.sessions.create_index("expires_at")
            
            # Forecast indexes
            await self.forecasts.create_index("user_id")
            await self.forecasts.create_index("created_at")
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to MongoDB"""
        return self.client is not None
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    # User operations
    async def create_user(self, username: str, email: str, password: str) -> Optional[Dict]:
        """Create a new user"""
        try:
            # Hash password (in production, use proper bcrypt)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user_doc = {
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "is_active": True
            }
            
            result = await self.users.insert_one(user_doc)
            
            # Return user without password
            user_doc["id"] = str(result.inserted_id)
            del user_doc["password_hash"]
            del user_doc["_id"]
            
            logger.info(f"User created: {username}")
            return user_doc
            
        except DuplicateKeyError:
            logger.warning(f"User already exists: {username} or {email}")
            return None
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return None
    
    async def get_user_by_credentials(self, username: str, password: str) -> Optional[Dict]:
        """Get user by username and password"""
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            user = await self.users.find_one({
                "username": username,
                "password_hash": password_hash,
                "is_active": True
            })
            
            if user:
                user["id"] = str(user["_id"])
                del user["password_hash"]
                del user["_id"]
                return user
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by credentials: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        try:
            from bson import ObjectId
            
            user = await self.users.find_one({
                "_id": ObjectId(user_id),
                "is_active": True
            })
            
            if user:
                user["id"] = str(user["_id"])
                del user["password_hash"]
                del user["_id"]
                return user
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None
    
    # Session operations
    async def create_session(self, user_id: str) -> str:
        """Create a new session token"""
        try:
            import secrets
            
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow().replace(hour=23, minute=59, second=59)  # End of day
            
            session_doc = {
                "token": token,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at
            }
            
            await self.sessions.insert_one(session_doc)
            logger.info(f"Session created for user: {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    async def get_user_by_token(self, token: str) -> Optional[Dict]:
        """Get user by session token"""
        try:
            session = await self.sessions.find_one({
                "token": token,
                "expires_at": {"$gt": datetime.utcnow()}
            })
            
            if session:
                user = await self.get_user_by_id(session["user_id"])
                return user
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by token: {e}")
            return None
    
    async def delete_session(self, token: str) -> bool:
        """Delete a session (logout)"""
        try:
            result = await self.sessions.delete_one({"token": token})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    # Forecast operations
    async def create_forecast(self, user_id: str, forecast_data: Dict) -> Optional[Dict]:
        """Create a new forecast record"""
        try:
            forecast_doc = {
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                **forecast_data
            }
            
            result = await self.forecasts.insert_one(forecast_doc)
            
            forecast_doc["id"] = str(result.inserted_id)
            del forecast_doc["_id"]
            
            logger.info(f"Forecast created for user: {user_id}")
            return forecast_doc
            
        except Exception as e:
            logger.error(f"Failed to create forecast: {e}")
            return None
    
    async def get_user_forecasts(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's forecasts"""
        try:
            cursor = self.forecasts.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit)
            
            forecasts = []
            async for forecast in cursor:
                forecast["id"] = str(forecast["_id"])
                del forecast["_id"]
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Failed to get user forecasts: {e}")
            return []
    
    async def get_forecast_by_id(self, forecast_id: str, user_id: str) -> Optional[Dict]:
        """Get specific forecast by ID"""
        try:
            from bson import ObjectId
            
            forecast = await self.forecasts.find_one({
                "_id": ObjectId(forecast_id),
                "user_id": user_id
            })
            
            if forecast:
                forecast["id"] = str(forecast["_id"])
                del forecast["_id"]
                return forecast
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get forecast by ID: {e}")
            return None
    
    # Dashboard operations
    async def get_dashboard_stats(self, user_id: str) -> Dict:
        """Get dashboard statistics for user — real metrics from forecast history"""
        try:
            # Count total forecasts
            total_forecasts = await self.forecasts.count_documents({"user_id": user_id})
            
            # Compute real averages using aggregation
            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": None,
                    "avg_coverage": {"$avg": "$cloud_coverage_pct"},
                    "avg_processing_time": {"$avg": "$processing_time"},
                }}
            ]
            
            avg_coverage = 0.0
            avg_processing_time = 0.0
            
            async for result in self.forecasts.aggregate(pipeline):
                avg_coverage = result.get("avg_coverage", 0.0) or 0.0
                avg_processing_time = result.get("avg_processing_time", 0.0) or 0.0
            
            # Get recent forecasts
            recent_cursor = self.forecasts.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(5)
            
            recent_forecasts = []
            async for forecast in recent_cursor:
                recent_forecasts.append({
                    "id": str(forecast["_id"]),
                    "location": forecast.get("location", "Satellite"),
                    "cloud_coverage_pct": forecast.get("cloud_coverage_pct", 0),
                    "model_type": forecast.get("model_type", "Unknown"),
                    "date": forecast["created_at"].isoformat()
                })
            
            return {
                "total_forecasts": total_forecasts,
                "avg_cloud_coverage": round(avg_coverage, 1),
                "avg_processing_time": round(avg_processing_time, 3),
                "recent_forecasts": recent_forecasts
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard stats: {e}")
            return {
                "total_forecasts": 0,
                "avg_cloud_coverage": 0.0,
                "avg_processing_time": 0.0,
                "recent_forecasts": []
            }

# Global MongoDB instance
mongodb = MongoDB()
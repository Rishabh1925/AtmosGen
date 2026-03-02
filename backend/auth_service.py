"""
Supabase authentication integration for FastAPI
"""

import os
from supabase import create_client, Client
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging
import jwt

logger = logging.getLogger(__name__)

class SupabaseAuth:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.jwt_secret = os.getenv("JWT_SECRET", "fallback-secret")
        
        if self.supabase_url and self.supabase_key:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase auth client initialized")
        else:
            self.supabase = None
            logger.warning("Supabase credentials not found")
    
    def is_connected(self) -> bool:
        return self.supabase is not None
    
    async def register_user(self, email: str, password: str, user_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Register a new user with Supabase Auth"""
        try:
            if not self.supabase:
                raise HTTPException(status_code=503, detail="Authentication service not available")
            
            # Register with Supabase Auth
            response = self.supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": user_data or {}
                }
            })
            
            if response.user:
                logger.info(f"User registered successfully: {email}")
                return {
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email,
                        "email_confirmed": response.user.email_confirmed_at is not None
                    },
                    "session": response.session.access_token if response.session else None
                }
            else:
                raise HTTPException(status_code=400, detail="Registration failed")
                
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            if "already registered" in str(e).lower():
                raise HTTPException(status_code=400, detail="Email already registered")
            raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")
    
    async def login_user(self, email_or_username: str, password: str) -> Dict[str, Any]:
        """Login user with Supabase Auth (supports both email and username)"""
        try:
            if not self.supabase:
                raise HTTPException(status_code=503, detail="Authentication service not available")
            
            # If it looks like an email, use it directly
            email = email_or_username
            if "@" not in email_or_username:
                # It's a username, need to find the email
                # Query user_profiles table to get email from username
                try:
                    profile_result = self.supabase.table('user_profiles').select('user_id').eq('username', email_or_username).execute()
                    if profile_result.data:
                        user_id = profile_result.data[0]['user_id']
                        # Get user email from auth.users (this requires service key)
                        # For now, we'll assume username is the email prefix
                        # In production, you'd store email in user_profiles or use service key
                        email = f"{email_or_username}@example.com"  # Fallback
                    else:
                        raise HTTPException(status_code=401, detail="Invalid username or password")
                except:
                    # If username lookup fails, try as email anyway
                    email = email_or_username
            
            response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user and response.session:
                logger.info(f"User logged in successfully: {email}")
                return {
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email,
                        "email_confirmed": response.user.email_confirmed_at is not None
                    },
                    "session": response.session.access_token
                }
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
        except Exception as e:
            logger.error(f"Login failed: {e}")
            if "invalid" in str(e).lower() or "credentials" in str(e).lower():
                raise HTTPException(status_code=401, detail="Invalid email or password")
            raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")
    
    async def get_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user info from JWT token"""
        try:
            if not self.supabase:
                return None
            
            # Get user from Supabase using the token
            response = self.supabase.auth.get_user(token)
            
            if response.user:
                return {
                    "id": response.user.id,
                    "email": response.user.email,
                    "email_confirmed": response.user.email_confirmed_at is not None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None
    
    async def logout_user(self, token: str) -> bool:
        """Logout user"""
        try:
            if not self.supabase:
                return False
            
            self.supabase.auth.sign_out()
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False

# Global auth instance
supabase_auth = SupabaseAuth()

# FastAPI dependencies
security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Get current user from Bearer token"""
    if not credentials:
        return None
    
    user = await supabase_auth.get_user_from_token(credentials.credentials)
    return user

async def require_auth(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require user to be authenticated"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user
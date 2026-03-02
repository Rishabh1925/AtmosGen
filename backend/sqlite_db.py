"""
SQLite database fallback for local development
"""

import sqlite3
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SQLiteDB:
    def __init__(self, db_path: str = "atmosgen.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        first_name TEXT,
                        last_name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        email_confirmed BOOLEAN DEFAULT FALSE
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS forecasts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        input_images_count INTEGER,
                        generated_image TEXT,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                conn.commit()
                logger.info("SQLite database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
    
    def is_connected(self) -> bool:
        """Check if database is available"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
                return True
        except:
            return False
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split(':')
            password_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_check.hex() == hash_hex
        except:
            return False
    
    def create_user(self, email: str, password: str, username: str, first_name: str = None, last_name: str = None) -> Optional[str]:
        """Create a new user"""
        try:
            user_id = secrets.token_urlsafe(16)
            password_hash = self.hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users (id, email, username, password_hash, first_name, last_name, email_confirmed)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, email, username, password_hash, first_name, last_name, True))
                conn.commit()
                
            logger.info(f"User created successfully: {email}")
            return user_id
            
        except sqlite3.IntegrityError as e:
            if "email" in str(e):
                logger.error("Email already exists")
                return None
            elif "username" in str(e):
                logger.error("Username already exists")
                return None
            else:
                logger.error(f"User creation failed: {e}")
                return None
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return None
    
    def authenticate_user(self, email_or_username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Try email first, then username
                user = conn.execute("""
                    SELECT * FROM users WHERE email = ? OR username = ?
                """, (email_or_username, email_or_username)).fetchone()
                
                if user and self.verify_password(password, user['password_hash']):
                    return {
                        'id': user['id'],
                        'email': user['email'],
                        'username': user['username'],
                        'first_name': user['first_name'],
                        'last_name': user['last_name'],
                        'email_confirmed': bool(user['email_confirmed'])
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    def create_session(self, user_id: str) -> str:
        """Create a session token for user"""
        try:
            session_token = secrets.token_urlsafe(32)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_sessions (session_token, user_id)
                    VALUES (?, ?)
                """, (session_token, user_id))
                conn.commit()
                
            return session_token
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return None
    
    def get_user_by_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get user by session token"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                result = conn.execute("""
                    SELECT u.* FROM users u
                    JOIN user_sessions s ON u.id = s.user_id
                    WHERE s.session_token = ?
                """, (session_token,)).fetchone()
                
                if result:
                    return {
                        'id': result['id'],
                        'email': result['email'],
                        'username': result['username'],
                        'first_name': result['first_name'],
                        'last_name': result['last_name'],
                        'email_confirmed': bool(result['email_confirmed'])
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Session lookup failed: {e}")
            return None
    
    def delete_session(self, session_token: str) -> bool:
        """Delete a session token"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Session deletion failed: {e}")
            return False
    
    def save_forecast(self, user_id: str, forecast_data: Dict[str, Any]) -> Optional[int]:
        """Save forecast to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO forecasts (user_id, name, description, input_images_count, generated_image, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    forecast_data.get('name', 'Untitled Forecast'),
                    forecast_data.get('description'),
                    forecast_data.get('input_images_count'),
                    forecast_data.get('generated_image'),
                    forecast_data.get('processing_time')
                ))
                conn.commit()
                return cursor.lastrowid
                
        except Exception as e:
            logger.error(f"Forecast save failed: {e}")
            return None
    
    def get_user_forecasts(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's forecast history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                results = conn.execute("""
                    SELECT * FROM forecasts WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (user_id, limit)).fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Get forecasts failed: {e}")
            return []
    
    def get_forecast_by_id(self, forecast_id: int, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific forecast by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                result = conn.execute("""
                    SELECT * FROM forecasts WHERE id = ? AND user_id = ?
                """, (forecast_id, user_id)).fetchone()
                
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Get forecast failed: {e}")
            return None
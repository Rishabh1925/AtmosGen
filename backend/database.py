import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "atmosgen.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)
                
                # Sessions table for simple session management
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        session_token TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # Forecasts table to save user forecasts
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS forecasts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        name TEXT,
                        input_images_count INTEGER NOT NULL,
                        generated_image TEXT NOT NULL,
                        processing_time REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
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
    
    def create_user(self, username: str, email: str, password: str) -> Optional[int]:
        """Create a new user"""
        try:
            password_hash = self.hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash)
                    VALUES (?, ?, ?)
                """, (username, email, password_hash))
                
                user_id = cursor.lastrowid
                conn.commit()
                logger.info(f"User created successfully: {username}")
                return user_id
                
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed - duplicate: {e}")
            return None
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, email, password_hash, is_active
                    FROM users 
                    WHERE username = ? OR email = ?
                """, (username, username))
                
                user = cursor.fetchone()
                
                if user and user[4] and self.verify_password(password, user[3]):
                    # Update last login
                    cursor.execute("""
                        UPDATE users SET last_login = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """, (user[0],))
                    conn.commit()
                    
                    return {
                        'id': user[0],
                        'username': user[1],
                        'email': user[2]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    def create_session(self, user_id: int) -> str:
        """Create a new session for user"""
        try:
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(days=7)  # 7 day expiry
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (user_id, session_token, expires_at)
                    VALUES (?, ?, ?)
                """, (user_id, session_token, expires_at))
                conn.commit()
                
            logger.info(f"Session created for user {user_id}")
            return session_token
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return None
    
    def get_user_by_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get user info by session token"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT u.id, u.username, u.email, s.expires_at
                    FROM users u
                    JOIN sessions s ON u.id = s.user_id
                    WHERE s.session_token = ? AND s.is_active = 1 AND s.expires_at > CURRENT_TIMESTAMP
                """, (session_token,))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'id': result[0],
                        'username': result[1],
                        'email': result[2]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    def logout_session(self, session_token: str) -> bool:
        """Logout by deactivating session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions SET is_active = 0 
                    WHERE session_token = ?
                """, (session_token,))
                conn.commit()
                
            logger.info("Session logged out successfully")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    def save_forecast(self, user_id: int, name: str, input_images_count: int, 
                     generated_image: str, processing_time: float) -> Optional[int]:
        """Save a forecast to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO forecasts (user_id, name, input_images_count, generated_image, processing_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, name, input_images_count, generated_image, processing_time))
                
                forecast_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Forecast saved for user {user_id}")
                return forecast_id
                
        except Exception as e:
            logger.error(f"Forecast save failed: {e}")
            return None
    
    def get_user_forecasts(self, user_id: int, limit: int = 50) -> list:
        """Get user's forecast history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, input_images_count, processing_time, created_at
                    FROM forecasts 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                forecasts = []
                for row in cursor.fetchall():
                    forecasts.append({
                        'id': row[0],
                        'name': row[1],
                        'input_images_count': row[2],
                        'processing_time': row[3],
                        'created_at': row[4]
                    })
                
                return forecasts
                
        except Exception as e:
            logger.error(f"Get forecasts failed: {e}")
            return []
    
    def get_forecast_by_id(self, forecast_id: int, user_id: int) -> Optional[Dict[str, Any]]:
        """Get specific forecast by ID (only if owned by user)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, input_images_count, generated_image, processing_time, created_at
                    FROM forecasts 
                    WHERE id = ? AND user_id = ?
                """, (forecast_id, user_id))
                
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'input_images_count': row[2],
                        'generated_image': row[3],
                        'processing_time': row[4],
                        'created_at': row[5]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Get forecast failed: {e}")
            return None
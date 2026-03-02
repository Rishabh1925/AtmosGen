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
                        first_name TEXT,
                        last_name TEXT,
                        bio TEXT,
                        avatar_url TEXT,
                        location TEXT,
                        timezone TEXT DEFAULT 'UTC',
                        email_notifications BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        is_admin BOOLEAN DEFAULT 0,
                        total_predictions INTEGER DEFAULT 0,
                        total_processing_time REAL DEFAULT 0.0
                    )
                """)
                
                # User preferences table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        theme TEXT DEFAULT 'light',
                        default_region TEXT DEFAULT 'north_america',
                        default_layer TEXT DEFAULT 'visible',
                        auto_save_forecasts BOOLEAN DEFAULT 1,
                        max_forecast_history INTEGER DEFAULT 100,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # User activity log
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_activity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        activity_type TEXT NOT NULL,
                        activity_data TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                
                # System stats table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stat_date DATE NOT NULL,
                        total_users INTEGER DEFAULT 0,
                        active_users INTEGER DEFAULT 0,
                        total_predictions INTEGER DEFAULT 0,
                        avg_processing_time REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                        description TEXT,
                        input_images_count INTEGER NOT NULL,
                        generated_image TEXT NOT NULL,
                        processing_time REAL NOT NULL,
                        model_version TEXT DEFAULT 'v1.0',
                        region TEXT,
                        layer TEXT,
                        source_type TEXT DEFAULT 'upload',
                        is_favorite BOOLEAN DEFAULT 0,
                        is_public BOOLEAN DEFAULT 0,
                        view_count INTEGER DEFAULT 0,
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
    
    def create_user(self, username: str, email: str, password: str, 
                   first_name: str = None, last_name: str = None) -> Optional[int]:
        """Create a new user with enhanced profile"""
        try:
            password_hash = self.hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, first_name, last_name)
                    VALUES (?, ?, ?, ?, ?)
                """, (username, email, password_hash, first_name, last_name))
                
                user_id = cursor.lastrowid
                
                # Create default user preferences
                cursor.execute("""
                    INSERT INTO user_preferences (user_id)
                    VALUES (?)
                """, (user_id,))
                
                # Log user registration activity
                cursor.execute("""
                    INSERT INTO user_activity (user_id, activity_type, activity_data)
                    VALUES (?, 'user_registered', ?)
                """, (user_id, f'{{"username": "{username}", "email": "{email}"}}'))
                
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
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get complete user profile with stats"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT u.id, u.username, u.email, u.first_name, u.last_name, 
                           u.bio, u.avatar_url, u.location, u.timezone, 
                           u.email_notifications, u.created_at, u.last_login,
                           u.total_predictions, u.total_processing_time,
                           p.theme, p.default_region, p.default_layer, 
                           p.auto_save_forecasts, p.max_forecast_history
                    FROM users u
                    LEFT JOIN user_preferences p ON u.id = p.user_id
                    WHERE u.id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                
                if row:
                    # Get additional stats
                    cursor.execute("""
                        SELECT COUNT(*) as forecast_count,
                               COUNT(CASE WHEN is_favorite = 1 THEN 1 END) as favorite_count,
                               AVG(processing_time) as avg_processing_time
                        FROM forecasts 
                        WHERE user_id = ?
                    """, (user_id,))
                    
                    stats = cursor.fetchone()
                    
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'first_name': row[3],
                        'last_name': row[4],
                        'bio': row[5],
                        'avatar_url': row[6],
                        'location': row[7],
                        'timezone': row[8],
                        'email_notifications': row[9],
                        'created_at': row[10],
                        'last_login': row[11],
                        'total_predictions': row[12] or 0,
                        'total_processing_time': row[13] or 0.0,
                        'preferences': {
                            'theme': row[14] or 'light',
                            'default_region': row[15] or 'north_america',
                            'default_layer': row[16] or 'visible',
                            'auto_save_forecasts': row[17] if row[17] is not None else True,
                            'max_forecast_history': row[18] or 100
                        },
                        'stats': {
                            'forecast_count': stats[0] or 0,
                            'favorite_count': stats[1] or 0,
                            'avg_processing_time': stats[2] or 0.0
                        }
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Get user profile failed: {e}")
            return None
    
    def update_user_profile(self, user_id: int, profile_data: Dict[str, Any]) -> bool:
        """Update user profile information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update user table
                user_fields = ['first_name', 'last_name', 'bio', 'avatar_url', 
                              'location', 'timezone', 'email_notifications']
                
                update_fields = []
                update_values = []
                
                for field in user_fields:
                    if field in profile_data:
                        update_fields.append(f"{field} = ?")
                        update_values.append(profile_data[field])
                
                if update_fields:
                    update_values.append(user_id)
                    cursor.execute(f"""
                        UPDATE users 
                        SET {', '.join(update_fields)}
                        WHERE id = ?
                    """, update_values)
                
                # Update preferences if provided
                if 'preferences' in profile_data:
                    prefs = profile_data['preferences']
                    pref_fields = ['theme', 'default_region', 'default_layer', 
                                  'auto_save_forecasts', 'max_forecast_history']
                    
                    pref_update_fields = []
                    pref_update_values = []
                    
                    for field in pref_fields:
                        if field in prefs:
                            pref_update_fields.append(f"{field} = ?")
                            pref_update_values.append(prefs[field])
                    
                    if pref_update_fields:
                        pref_update_values.extend([user_id, user_id])
                        cursor.execute(f"""
                            INSERT OR REPLACE INTO user_preferences 
                            (user_id, {', '.join(pref_fields)})
                            VALUES (?, {', '.join(['?' for _ in pref_fields])})
                        """, [user_id] + [prefs.get(field) for field in pref_fields])
                
                # Log activity
                cursor.execute("""
                    INSERT INTO user_activity (user_id, activity_type, activity_data)
                    VALUES (?, 'profile_updated', ?)
                """, (user_id, str(profile_data)))
                
                conn.commit()
                logger.info(f"User profile updated for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Update user profile failed: {e}")
            return False
    
    def update_user_stats(self, user_id: int, processing_time: float) -> bool:
        """Update user statistics after a prediction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users 
                    SET total_predictions = total_predictions + 1,
                        total_processing_time = total_processing_time + ?
                    WHERE id = ?
                """, (processing_time, user_id))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Update user stats failed: {e}")
            return False
    
    def get_user_activity(self, user_id: int, limit: int = 20) -> list:
        """Get user activity history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT activity_type, activity_data, created_at
                    FROM user_activity 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                activities = []
                for row in cursor.fetchall():
                    activities.append({
                        'activity_type': row[0],
                        'activity_data': row[1],
                        'created_at': row[2]
                    })
                
                return activities
                
        except Exception as e:
            logger.error(f"Get user activity failed: {e}")
            return []
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current password hash
                cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
                result = cursor.fetchone()
                
                if not result or not self.verify_password(old_password, result[0]):
                    return False
                
                # Update password
                new_hash = self.hash_password(new_password)
                cursor.execute("""
                    UPDATE users SET password_hash = ? WHERE id = ?
                """, (new_hash, user_id))
                
                # Log activity
                cursor.execute("""
                    INSERT INTO user_activity (user_id, activity_type)
                    VALUES (?, 'password_changed')
                """, (user_id,))
                
                conn.commit()
                logger.info(f"Password changed for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Change password failed: {e}")
            return False
    
    def get_dashboard_stats(self, user_id: int) -> Dict[str, Any]:
        """Get dashboard statistics for user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # User's forecast stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_forecasts,
                        COUNT(CASE WHEN DATE(created_at) = DATE('now') THEN 1 END) as today_forecasts,
                        COUNT(CASE WHEN DATE(created_at) >= DATE('now', '-7 days') THEN 1 END) as week_forecasts,
                        AVG(processing_time) as avg_processing_time,
                        MAX(created_at) as last_forecast
                    FROM forecasts 
                    WHERE user_id = ?
                """, (user_id,))
                
                forecast_stats = cursor.fetchone()
                
                # Recent forecasts
                cursor.execute("""
                    SELECT id, name, processing_time, created_at
                    FROM forecasts 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT 5
                """, (user_id,))
                
                recent_forecasts = []
                for row in cursor.fetchall():
                    recent_forecasts.append({
                        'id': row[0],
                        'name': row[1],
                        'processing_time': row[2],
                        'created_at': row[3]
                    })
                
                return {
                    'total_forecasts': forecast_stats[0] or 0,
                    'today_forecasts': forecast_stats[1] or 0,
                    'week_forecasts': forecast_stats[2] or 0,
                    'avg_processing_time': forecast_stats[3] or 0.0,
                    'last_forecast': forecast_stats[4],
                    'recent_forecasts': recent_forecasts
                }
                
        except Exception as e:
            logger.error(f"Get dashboard stats failed: {e}")
            return {}
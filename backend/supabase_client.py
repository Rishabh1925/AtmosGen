"""
Supabase client configuration and database operations
"""

import os
from supabase import create_client, Client
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class SupabaseDB:
    def __init__(self):
        # Get Supabase credentials from environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not found. Using local SQLite fallback.")
            self.supabase = None
            return
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.supabase = None
    
    def is_connected(self) -> bool:
        """Check if Supabase is connected"""
        return self.supabase is not None
    
    async def create_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Create user profile in Supabase"""
        try:
            if not self.supabase:
                return False
            
            result = self.supabase.table('user_profiles').insert({
                'user_id': user_id,
                'username': profile_data.get('username'),
                'first_name': profile_data.get('first_name'),
                'last_name': profile_data.get('last_name'),
                'bio': profile_data.get('bio'),
                'location': profile_data.get('location'),
                'timezone': profile_data.get('timezone', 'UTC'),
                'email_notifications': profile_data.get('email_notifications', True),
                'total_predictions': 0,
                'total_processing_time': 0.0
            }).execute()
            
            # Create default preferences
            self.supabase.table('user_preferences').insert({
                'user_id': user_id,
                'theme': 'light',
                'default_region': 'north_america',
                'default_layer': 'visible',
                'auto_save_forecasts': True,
                'max_forecast_history': 100
            }).execute()
            
            logger.info(f"User profile created for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile with stats and preferences"""
        try:
            if not self.supabase:
                return None
            
            # Get user profile
            profile_result = self.supabase.table('user_profiles').select('*').eq('user_id', user_id).execute()
            
            if not profile_result.data:
                return None
            
            profile = profile_result.data[0]
            
            # Get preferences
            prefs_result = self.supabase.table('user_preferences').select('*').eq('user_id', user_id).execute()
            preferences = prefs_result.data[0] if prefs_result.data else {}
            
            # Get forecast stats
            forecasts_result = self.supabase.table('forecasts').select('*').eq('user_id', user_id).execute()
            forecasts = forecasts_result.data or []
            
            # Calculate stats
            total_forecasts = len(forecasts)
            favorite_count = len([f for f in forecasts if f.get('is_favorite')])
            avg_processing_time = sum(f.get('processing_time', 0) for f in forecasts) / total_forecasts if total_forecasts > 0 else 0
            
            return {
                **profile,
                'preferences': preferences,
                'stats': {
                    'forecast_count': total_forecasts,
                    'favorite_count': favorite_count,
                    'avg_processing_time': avg_processing_time
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    async def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            if not self.supabase:
                return False
            
            # Update profile
            if any(key in profile_data for key in ['first_name', 'last_name', 'bio', 'location', 'timezone', 'email_notifications']):
                profile_updates = {k: v for k, v in profile_data.items() 
                                 if k in ['first_name', 'last_name', 'bio', 'location', 'timezone', 'email_notifications']}
                
                self.supabase.table('user_profiles').update(profile_updates).eq('user_id', user_id).execute()
            
            # Update preferences
            if 'preferences' in profile_data:
                prefs = profile_data['preferences']
                self.supabase.table('user_preferences').update(prefs).eq('user_id', user_id).execute()
            
            logger.info(f"User profile updated for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
            return False
    
    async def save_forecast(self, user_id: str, forecast_data: Dict[str, Any]) -> Optional[str]:
        """Save forecast to Supabase"""
        try:
            if not self.supabase:
                return None
            
            result = self.supabase.table('forecasts').insert({
                'user_id': user_id,
                'name': forecast_data.get('name'),
                'description': forecast_data.get('description'),
                'input_images_count': forecast_data.get('input_images_count'),
                'generated_image': forecast_data.get('generated_image'),
                'processing_time': forecast_data.get('processing_time'),
                'model_version': forecast_data.get('model_version', 'v1.0'),
                'region': forecast_data.get('region'),
                'layer': forecast_data.get('layer'),
                'source_type': forecast_data.get('source_type', 'upload'),
                'is_favorite': False,
                'is_public': False,
                'view_count': 0
            }).execute()
            
            if result.data:
                forecast_id = result.data[0]['id']
                
                # Update user stats
                await self.update_user_stats(user_id, forecast_data.get('processing_time', 0))
                
                logger.info(f"Forecast saved with ID {forecast_id}")
                return forecast_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to save forecast: {e}")
            return None
    
    async def get_user_forecasts(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's forecast history"""
        try:
            if not self.supabase:
                return []
            
            result = self.supabase.table('forecasts').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get user forecasts: {e}")
            return []
    
    async def get_forecast_by_id(self, forecast_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific forecast by ID"""
        try:
            if not self.supabase:
                return None
            
            result = self.supabase.table('forecasts').select('*').eq('id', forecast_id).eq('user_id', user_id).execute()
            
            if result.data:
                # Increment view count
                self.supabase.table('forecasts').update({
                    'view_count': result.data[0]['view_count'] + 1
                }).eq('id', forecast_id).execute()
                
                return result.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get forecast: {e}")
            return None
    
    async def update_user_stats(self, user_id: str, processing_time: float) -> bool:
        """Update user statistics"""
        try:
            if not self.supabase:
                return False
            
            # Get current stats
            result = self.supabase.table('user_profiles').select('total_predictions', 'total_processing_time').eq('user_id', user_id).execute()
            
            if result.data:
                current_stats = result.data[0]
                new_total_predictions = current_stats['total_predictions'] + 1
                new_total_time = current_stats['total_processing_time'] + processing_time
                
                # Update stats
                self.supabase.table('user_profiles').update({
                    'total_predictions': new_total_predictions,
                    'total_processing_time': new_total_time
                }).eq('user_id', user_id).execute()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update user stats: {e}")
            return False
    
    async def get_dashboard_stats(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard statistics"""
        try:
            if not self.supabase:
                return {}
            
            # Get all forecasts for stats calculation
            forecasts_result = self.supabase.table('forecasts').select('*').eq('user_id', user_id).execute()
            forecasts = forecasts_result.data or []
            
            from datetime import datetime, timedelta
            now = datetime.now()
            today = now.date()
            week_ago = now - timedelta(days=7)
            
            # Calculate stats
            total_forecasts = len(forecasts)
            today_forecasts = len([f for f in forecasts if f['created_at'][:10] == str(today)])
            week_forecasts = len([f for f in forecasts if f['created_at'] >= week_ago.isoformat()])
            avg_processing_time = sum(f['processing_time'] for f in forecasts) / total_forecasts if total_forecasts > 0 else 0
            
            # Recent forecasts
            recent_forecasts = sorted(forecasts, key=lambda x: x['created_at'], reverse=True)[:5]
            
            return {
                'total_forecasts': total_forecasts,
                'today_forecasts': today_forecasts,
                'week_forecasts': week_forecasts,
                'avg_processing_time': avg_processing_time,
                'last_forecast': recent_forecasts[0]['created_at'] if recent_forecasts else None,
                'recent_forecasts': recent_forecasts
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard stats: {e}")
            return {}
    
    async def log_user_activity(self, user_id: str, activity_type: str, activity_data: str = None) -> bool:
        """Log user activity"""
        try:
            if not self.supabase:
                return False
            
            self.supabase.table('user_activity').insert({
                'user_id': user_id,
                'activity_type': activity_type,
                'activity_data': activity_data
            }).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log user activity: {e}")
            return False
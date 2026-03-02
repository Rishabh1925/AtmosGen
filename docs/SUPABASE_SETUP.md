# Supabase Setup Guide for AtmosGen

## Why Supabase?

**Advantages over SQLite:**
- ✅ Cloud-hosted PostgreSQL database
- ✅ Built-in authentication (no custom auth needed)
- ✅ Real-time subscriptions
- ✅ Automatic backups
- ✅ Easy to scale
- ✅ Great dashboard for data management
- ✅ Row Level Security (RLS)
- ✅ Free tier with generous limits

## Setup Steps

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Sign up/Login with GitHub
4. Click "New Project"
5. Choose organization and project name: `atmosgen`
6. Set database password (save this!)
7. Choose region closest to you
8. Click "Create new project"

### 2. Get API Keys

1. In your Supabase dashboard, go to Settings > API
2. Copy these values:
   - **Project URL**: `https://your-project.supabase.co`
   - **Anon public key**: `eyJ...` (starts with eyJ)
   - **Service role key**: `eyJ...` (keep this secret!)

### 3. Set up Database Schema

1. In Supabase dashboard, go to SQL Editor
2. Copy the contents of `backend/supabase_schema.sql`
3. Paste and run the SQL script
4. This creates all tables with proper relationships and security

### 4. Configure Environment Variables

1. Copy `backend/.env.example` to `backend/.env`
2. Fill in your Supabase credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-key-here
PORT=8000
ENVIRONMENT=development
JWT_SECRET=your-random-jwt-secret
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 5. Enable Authentication

1. In Supabase dashboard, go to Authentication > Settings
2. Enable these providers:
   - ✅ Email (enabled by default)
   - ✅ Google (optional)
   - ✅ GitHub (optional)

3. Configure email templates (optional):
   - Go to Authentication > Email Templates
   - Customize signup confirmation, password reset emails

### 6. Set up Row Level Security

The SQL schema already includes RLS policies, but verify:

1. Go to Authentication > Policies
2. You should see policies for:
   - `user_profiles` - Users can only access their own data
   - `user_preferences` - Users can only access their own preferences
   - `forecasts` - Users can access their own + public forecasts
   - `user_activity` - Users can only see their own activity

### 7. Test Database Connection

```bash
cd backend
pip install -r requirements.txt
python -c "
from supabase_client import SupabaseDB
import os
os.environ['SUPABASE_URL'] = 'your-url'
os.environ['SUPABASE_ANON_KEY'] = 'your-key'

db = SupabaseDB()
print('Connected:', db.is_connected())
"
```

## Database Schema Overview

### Tables Created:

1. **user_profiles** - Extended user information
   - Links to Supabase auth.users
   - Stores username, name, bio, location, stats

2. **user_preferences** - User settings
   - Theme, default region/layer
   - Auto-save preferences, history limits

3. **forecasts** - Weather predictions
   - Generated images, processing times
   - Metadata (region, layer, model version)
   - Favorite/public flags, view counts

4. **user_activity** - Activity logging
   - All user actions with timestamps
   - IP address and user agent tracking

5. **system_stats** - System analytics
   - Daily stats for admin dashboard

### Security Features:

- **Row Level Security (RLS)** - Users can only access their own data
- **Automatic user profile creation** - Triggered when user signs up
- **Secure API keys** - Anon key for client, service key for server
- **JWT-based authentication** - Handled by Supabase

## Migration from SQLite

The new system will:
1. Use Supabase Auth instead of custom authentication
2. Store all data in PostgreSQL instead of local SQLite
3. Enable real-time features (live dashboard updates)
4. Provide better security with RLS
5. Allow easy scaling and backups

## Next Steps

✅ **COMPLETED**: Database schema created in Supabase
✅ **COMPLETED**: Backend updated to use Supabase
✅ **COMPLETED**: Authentication system migrated

### Remaining Tasks:

1. **Update your `.env` file** with real Supabase credentials:
   ```env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your-anon-key-here
   SUPABASE_SERVICE_KEY=your-service-key-here
   ```

2. **Test the backend**:
   ```bash
   cd backend
   python main.py
   ```

3. **Fix frontend authentication bug**:
   - Frontend needs to send auth tokens properly
   - Check if cookies are being set/sent
   - Verify CORS configuration

4. **Deploy with Supabase configuration**

### Authentication Flow Fixed

The system now supports both:
- **Cookie-based auth**: Automatic for browsers
- **Bearer token auth**: For API clients

Frontend should check `/auth/me` endpoint to verify login status.

## Free Tier Limits

Supabase free tier includes:
- 500MB database storage
- 2GB bandwidth per month
- 50,000 monthly active users
- Unlimited API requests

Perfect for development and initial production deployment!
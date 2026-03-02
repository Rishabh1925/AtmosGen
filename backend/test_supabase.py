#!/usr/bin/env python3
"""
Test script to verify Supabase integration
Run this after setting up your .env file with real Supabase credentials
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from supabase_client import SupabaseDB
from auth_service import supabase_auth

async def test_supabase_connection():
    """Test basic Supabase connection and operations"""
    print("Testing Supabase Integration...")
    print("=" * 50)
    
    # Test database connection
    print("1. Testing database connection...")
    db = SupabaseDB()
    if db.is_connected():
        print("✅ Database connected successfully!")
    else:
        print("❌ Database connection failed. Check your SUPABASE_URL and SUPABASE_ANON_KEY")
        return False
    
    # Test auth connection
    print("\n2. Testing auth connection...")
    if supabase_auth.is_connected():
        print("✅ Auth service connected successfully!")
    else:
        print("❌ Auth service connection failed. Check your Supabase credentials")
        return False
    
    # Test user registration (optional - will create a test user)
    print("\n3. Testing user operations...")
    test_email = "test@example.com"
    test_password = "testpassword123"
    
    try:
        # Try to register a test user
        print(f"   Attempting to register test user: {test_email}")
        auth_result = await supabase_auth.register_user(
            test_email, 
            test_password,
            {"username": "testuser", "first_name": "Test", "last_name": "User"}
        )
        print("✅ User registration successful!")
        
        # Create user profile
        user_id = auth_result["user"]["id"]
        profile_created = await db.create_user_profile(user_id, {
            "username": "testuser",
            "first_name": "Test", 
            "last_name": "User"
        })
        
        if profile_created:
            print("✅ User profile created successfully!")
        else:
            print("⚠️  User profile creation failed (might already exist)")
        
    except Exception as e:
        if "already registered" in str(e):
            print("⚠️  Test user already exists (this is fine)")
        else:
            print(f"❌ User registration failed: {e}")
    
    print("\n" + "=" * 50)
    print("Supabase integration test completed!")
    print("\nIf you see ✅ for database and auth connections, you're ready to go!")
    print("Update your .env file with real credentials if you see ❌")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_supabase_connection())
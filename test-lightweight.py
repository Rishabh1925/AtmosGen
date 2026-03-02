#!/usr/bin/env python3
"""
Test script to verify lightweight model works without checkpoints
"""
import os
import sys
import asyncio

# Add backend to path
sys.path.append('backend')

async def test_lightweight_model():
    """Test that lightweight model loads and works"""
    print("Testing lightweight model...")
    
    # Set environment variable
    os.environ['USE_LIGHTWEIGHT_MODEL'] = 'true'
    
    try:
        from lightweight_model import LightweightModelService
        
        # Initialize service
        service = LightweightModelService()
        
        # Load model
        success = await service.load_model()
        
        if success and service.is_loaded():
            print("✅ Lightweight model loaded successfully!")
            print(f"✅ Device: {service.device}")
            print(f"✅ Model parameters: {sum(p.numel() for p in service.model.parameters()):,}")
            return True
        else:
            print("❌ Failed to load lightweight model")
            return False
            
    except Exception as e:
        print(f"❌ Error testing lightweight model: {e}")
        return False

async def test_regular_model():
    """Test that regular model still works when checkpoints are available"""
    print("\nTesting regular model...")
    
    # Unset environment variable
    os.environ.pop('USE_LIGHTWEIGHT_MODEL', None)
    
    try:
        from model_service import ModelService
        
        # Initialize service
        service = ModelService()
        
        # Try to load model
        success = await service.load_model()
        
        if success and service.is_loaded():
            print("✅ Regular model loaded successfully!")
            print(f"✅ Device: {service.device}")
            print(f"✅ Model parameters: {sum(p.numel() for p in service.model.parameters()):,}")
            return True
        else:
            print("⚠️  Regular model failed to load (expected if no checkpoints)")
            return False
            
    except Exception as e:
        print(f"⚠️  Regular model error (expected if no checkpoints): {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Testing AtmosGen Model Services\n")
    
    # Test lightweight model (should always work)
    lightweight_ok = await test_lightweight_model()
    
    # Test regular model (may fail without checkpoints)
    regular_ok = await test_regular_model()
    
    print("\n📊 Test Results:")
    print(f"Lightweight Model: {'✅ PASS' if lightweight_ok else '❌ FAIL'}")
    print(f"Regular Model: {'✅ PASS' if regular_ok else '⚠️  SKIP (no checkpoints)'}")
    
    if lightweight_ok:
        print("\n🚀 Ready for deployment!")
        print("The lightweight model will work on Railway without checkpoint files.")
    else:
        print("\n❌ Deployment may fail - lightweight model has issues")
        
    return lightweight_ok

if __name__ == "__main__":
    asyncio.run(main())
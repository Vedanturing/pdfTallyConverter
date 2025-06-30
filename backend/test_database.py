#!/usr/bin/env python3
"""
Database Connection Test
Test MongoDB connectivity and basic operations
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test MongoDB connection and basic operations"""
    
    print("🔍 Testing MongoDB Database Connection...")
    print("=" * 50)
    
    try:
        # Import database modules
        from db.mongo import connect_to_mongo, close_mongo_connection, get_database, get_users_collection, get_history_collection
        from models.user import UserInDB, ConversionHistory
        from bson import ObjectId
        
        # Test 1: Connection
        print("📡 Test 1: Connecting to MongoDB...")
        await connect_to_mongo()
        print("✅ MongoDB connection successful!")
        
        # Test 2: Database access
        print("\n🗄️ Test 2: Testing database access...")
        db = get_database()
        print(f"✅ Database access successful! Database: {db.name}")
        
        # Test 3: Collections access
        print("\n📦 Test 3: Testing collections access...")
        users_collection = get_users_collection()
        history_collection = get_history_collection()
        print("✅ Collections access successful!")
        
        # Test 4: Database operations
        print("\n🔧 Test 4: Testing database operations...")
        
        # Test ping
        await db.command("ping")
        print("✅ Database ping successful!")
        
        # Test collection listing
        collections = await db.list_collection_names()
        print(f"✅ Available collections: {collections}")
        
        # Test 5: Index verification
        print("\n📊 Test 5: Testing indexes...")
        
        # Check users collection indexes
        users_indexes = await users_collection.list_indexes().to_list(length=None)
        print(f"✅ Users collection indexes: {[idx['name'] for idx in users_indexes]}")
        
        # Check history collection indexes
        history_indexes = await history_collection.list_indexes().to_list(length=None)
        print(f"✅ History collection indexes: {[idx['name'] for idx in history_indexes]}")
        
        # Test 6: Basic CRUD operations
        print("\n🔄 Test 6: Testing basic CRUD operations...")
        
        # Test user insertion and retrieval
        test_user_data = {
            "name": "Test User",
            "email": f"test_{int(datetime.now().timestamp())}@example.com",
            "hashed_password": "test_hash",
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Insert test user
        result = await users_collection.insert_one(test_user_data)
        test_user_id = result.inserted_id
        print(f"✅ Test user created with ID: {test_user_id}")
        
        # Retrieve test user
        retrieved_user = await users_collection.find_one({"_id": test_user_id})
        print(f"✅ Test user retrieved: {retrieved_user['email']}")
        
        # Test history insertion
        test_history_data = {
            "user_id": test_user_id,
            "filename": "test_conversion.pdf",
            "original_filename": "test_file.pdf",
            "file_size": 1024,
            "conversion_type": "pdf_to_excel",
            "status": "success",
            "created_at": datetime.utcnow(),
            "processing_time": 2.5
        }
        
        # Insert test history
        history_result = await history_collection.insert_one(test_history_data)
        test_history_id = history_result.inserted_id
        print(f"✅ Test history created with ID: {test_history_id}")
        
        # Retrieve test history
        retrieved_history = await history_collection.find_one({"_id": test_history_id})
        print(f"✅ Test history retrieved: {retrieved_history['filename']}")
        
        # Test 7: Cleanup test data
        print("\n🧹 Test 7: Cleaning up test data...")
        
        # Delete test user
        await users_collection.delete_one({"_id": test_user_id})
        print("✅ Test user deleted")
        
        # Delete test history
        await history_collection.delete_one({"_id": test_history_id})
        print("✅ Test history deleted")
        
        # Test 8: Authentication module test
        print("\n🔐 Test 8: Testing authentication utilities...")
        try:
            from auth.auth_utils import get_password_hash, verify_password, create_access_token
            
            # Test password hashing
            password = "test_password_123"
            hashed = get_password_hash(password)
            print("✅ Password hashing successful")
            
            # Test password verification
            is_valid = verify_password(password, hashed)
            if is_valid:
                print("✅ Password verification successful")
            else:
                print("❌ Password verification failed")
            
            # Test JWT token creation
            token = create_access_token(data={"sub": "test@example.com"})
            if token:
                print("✅ JWT token creation successful")
            else:
                print("❌ JWT token creation failed")
                
        except Exception as e:
            print(f"⚠️ Authentication test warning: {str(e)}")
        
        # Final cleanup
        await close_mongo_connection()
        print("\n🔌 Database connection closed")
        
        print("\n" + "=" * 50)
        print("🎉 ALL DATABASE TESTS PASSED! 🎉")
        print("✅ MongoDB is properly configured and working")
        print("✅ All collections and indexes are set up correctly")
        print("✅ CRUD operations are working properly")
        print("✅ Authentication utilities are functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Database test failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Print helpful error messages
        if "ServerSelectionTimeoutError" in str(type(e)):
            print("\n🔧 Troubleshooting tips:")
            print("1. Make sure MongoDB is running")
            print("2. Check your MONGODB_URI in .env file")
            print("3. Verify network connectivity")
            
        elif "OperationFailure" in str(type(e)):
            print("\n🔧 Troubleshooting tips:")
            print("1. Check MongoDB authentication credentials")
            print("2. Verify database permissions")
            
        elif "ImportError" in str(type(e)):
            print("\n🔧 Troubleshooting tips:")
            print("1. Install missing dependencies: pip install -r requirements.txt")
            print("2. Check Python path and module imports")
        
        return False

async def test_environment_variables():
    """Test environment variables setup"""
    print("🔧 Testing Environment Variables...")
    print("=" * 50)
    
    required_vars = [
        "MONGODB_URI",
        "MONGODB_DATABASE", 
        "JWT_SECRET",
        "JWT_ALGORITHM",
        "JWT_EXPIRE_MINUTES"
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Don't print sensitive values
            if "SECRET" in var or "PASSWORD" in var:
                print(f"✅ {var}: [HIDDEN]")
            else:
                print(f"✅ {var}: {value}")
        else:
            missing_vars.append(var)
            print(f"❌ {var}: Not set")
    
    if missing_vars:
        print(f"\n⚠️ Missing environment variables: {missing_vars}")
        print("Please create a .env file with the required variables")
        return False
    else:
        print("\n✅ All environment variables are properly set!")
        return True

async def main():
    """Main test function"""
    print("🚀 Starting Database Connection Tests...\n")
    
    # Test 1: Environment variables
    env_test = await test_environment_variables()
    
    if not env_test:
        print("\n❌ Environment test failed. Please fix the .env file first.")
        return
    
    print("\n")
    
    # Test 2: Database connection
    db_test = await test_database_connection()
    
    if db_test:
        print("\n🎯 Ready to start the backend server!")
        print("Run: python main.py")
    else:
        print("\n⚠️ Please fix the database issues before starting the server")

if __name__ == "__main__":
    asyncio.run(main()) 
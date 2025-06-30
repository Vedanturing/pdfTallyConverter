import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    database = None

mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb.client = AsyncIOMotorClient(
            os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            maxPoolSize=10,
            minPoolSize=1,
            maxIdleTimeMS=45000,
            serverSelectionTimeoutMS=5000
        )
        
        # Get database
        database_name = os.getenv("MONGODB_DATABASE", "pdf_tally_converter")
        mongodb.database = mongodb.client[database_name]
        
        # Test connection
        await mongodb.client.admin.command('ping')
        logger.info(f"Connected to MongoDB database: {database_name}")
        
        # Setup indexes
        await setup_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise e

async def close_mongo_connection():
    """Close database connection"""
    if mongodb.client:
        mongodb.client.close()
        logger.info("MongoDB connection closed")

async def setup_indexes():
    """Create database indexes for better performance"""
    try:
        # Users collection indexes
        users_collection = mongodb.database.users
        await users_collection.create_index([("email", ASCENDING)], unique=True)
        await users_collection.create_index([("created_at", ASCENDING)])
        
        # Conversion history indexes
        history_collection = mongodb.database.conversion_history
        await history_collection.create_index([("user_id", ASCENDING)])
        await history_collection.create_index([("created_at", ASCENDING)])
        await history_collection.create_index([("user_id", ASCENDING), ("created_at", -1)])
        
        # TTL index for automatic cleanup after 30 days
        await history_collection.create_index(
            [("created_at", ASCENDING)], 
            expireAfterSeconds=30 * 24 * 60 * 60  # 30 days
        )
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.warning(f"Failed to create indexes: {str(e)}")

def get_database():
    """Get the database instance"""
    if mongodb.database is None:
        raise Exception("Database not connected. Call connect_to_mongo() first.")
    return mongodb.database

def get_collection(collection_name: str):
    """Get a specific collection"""
    database = get_database()
    return database[collection_name]

# Collection getters for convenience
def get_users_collection():
    """Get users collection"""
    return get_collection("users")

def get_history_collection():
    """Get conversion history collection"""
    return get_collection("conversion_history")

def get_sessions_collection():
    """Get sessions collection"""
    return get_collection("sessions") 
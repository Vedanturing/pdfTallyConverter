import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from bson import ObjectId

from models.user import UserCreate, UserLogin, UserResponse, Token, UserInDB, UserUpdate
from auth.auth_utils import (
    get_password_hash, 
    authenticate_user, 
    create_access_token,
    get_current_active_user,
    get_user_by_email
)
from db.mongo import get_users_collection, get_history_collection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/signup", response_model=dict)
async def signup(user_data: UserCreate):
    """Register a new user"""
    try:
        users_collection = get_users_collection()
        
        # Check if user already exists
        existing_user = await users_collection.find_one({"email": user_data.email})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password and create user
        hashed_password = get_password_hash(user_data.password)
        user_dict = {
            "name": user_data.name,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Insert user into database
        result = await users_collection.insert_one(user_dict)
        
        # Create access token
        access_token_expires = timedelta(minutes=1440)  # 24 hours
        access_token = create_access_token(
            data={"sub": user_data.email}, 
            expires_delta=access_token_expires
        )
        
        # Get the created user for response
        created_user = await users_collection.find_one({"_id": result.inserted_id})
        
        # Convert ObjectId to string for response
        user_response = UserResponse(
            id=str(created_user["_id"]),
            name=created_user["name"],
            email=created_user["email"],
            created_at=created_user["created_at"],
            is_active=created_user["is_active"]
        )
        
        logger.info(f"New user registered: {user_data.email}")
        
        return {
            "message": "User registered successfully",
            "user": user_response.model_dump(),
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )

@router.post("/login", response_model=dict)
async def login(user_credentials: UserLogin):
    """Authenticate user and return access token"""
    try:
        # Authenticate user
        user = await authenticate_user(user_credentials.email, user_credentials.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User account is deactivated"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=1440)  # 24 hours
        access_token = create_access_token(
            data={"sub": user.email}, 
            expires_delta=access_token_expires
        )
        
        # Prepare user response (without password)
        user_response = UserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            created_at=user.created_at,
            is_active=user.is_active
        )
        
        logger.info(f"User logged in: {user.email}")
        
        return {
            "message": "Login successful",
            "user": user_response.model_dump(),
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )

@router.post("/logout")
async def logout(current_user: UserInDB = Depends(get_current_active_user)):
    """Logout user (for completeness - JWT is stateless)"""
    try:
        logger.info(f"User logged out: {current_user.email}")
        return {"message": "Logout successful"}
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserInDB = Depends(get_current_active_user)):
    """Get current user information"""
    try:
        return UserResponse(
            id=current_user.id,
            name=current_user.name,
            email=current_user.email,
            created_at=current_user.created_at,
            is_active=current_user.is_active
        )
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting user information"
        )

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Update current user information"""
    try:
        users_collection = get_users_collection()
        
        # Prepare update data
        update_data = {}
        if user_update.name is not None:
            update_data["name"] = user_update.name
        if user_update.email is not None:
            # Check if email is already taken by another user
            existing_user = await users_collection.find_one({
                "email": user_update.email,
                "_id": {"$ne": current_user.id}
            })
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered by another user"
                )
            update_data["email"] = user_update.email
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided for update"
            )
        
        # Update user
        await users_collection.update_one(
            {"_id": current_user.id},
            {"$set": update_data}
        )
        
        # Get updated user
        updated_user = await users_collection.find_one({"_id": current_user.id})
        
        logger.info(f"User updated: {current_user.email}")
        
        return UserResponse(
            id=str(updated_user["_id"]),
            name=updated_user["name"],
            email=updated_user["email"],
            created_at=updated_user["created_at"],
            is_active=updated_user["is_active"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during user update"
        )

@router.delete("/delete_account")
async def delete_account(current_user: UserInDB = Depends(get_current_active_user)):
    """Delete user account and all associated data"""
    try:
        users_collection = get_users_collection()
        history_collection = get_history_collection()
        
        # Delete user's conversion history
        await history_collection.delete_many({"user_id": current_user.id})
        
        # Delete user account
        await users_collection.delete_one({"_id": current_user.id})
        
        logger.info(f"User account deleted: {current_user.email}")
        
        return {"message": "Account and all associated data deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting account: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during account deletion"
        ) 
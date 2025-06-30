import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends, Query
from bson import ObjectId

from models.user import UserInDB, ConversionHistory, ConversionHistoryResponse
from auth.auth_utils import get_optional_current_user
from db.mongo import get_history_collection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user_history"])

@router.get("/history", response_model=List[ConversionHistoryResponse])
async def get_user_history(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """Get user's conversion history"""
    try:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to view history"
            )
        
        history_collection = get_history_collection()
        
        # Get user's history with pagination
        cursor = history_collection.find(
            {"user_id": current_user.id}
        ).sort("created_at", -1).skip(offset).limit(limit)
        
        history_items = []
        async for item in cursor:
            history_items.append(ConversionHistoryResponse(
                id=str(item["_id"]),
                filename=item["filename"],
                original_filename=item["original_filename"],
                file_size=item["file_size"],
                conversion_type=item["conversion_type"],
                status=item["status"],
                created_at=item["created_at"],
                processing_time=item.get("processing_time")
            ))
        
        logger.info(f"Retrieved {len(history_items)} history items for user {current_user.email}")
        return history_items
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting history"
        )

@router.get("/history/{history_id}")
async def get_history_item_details(
    history_id: str,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """Get detailed information about a specific conversion"""
    try:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to view history details"
            )
        
        history_collection = get_history_collection()
        
        # Get the specific history item
        item = await history_collection.find_one({
            "_id": ObjectId(history_id),
            "user_id": current_user.id
        })
        
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="History item not found"
            )
        
        # Convert ObjectId to string for JSON response
        item["_id"] = str(item["_id"])
        item["user_id"] = str(item["user_id"])
        
        return item
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history item details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error getting history details"
        )

@router.delete("/history/{history_id}")
async def delete_history_item(
    history_id: str,
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """Delete a specific conversion from history"""
    try:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to delete history"
            )
        
        history_collection = get_history_collection()
        
        # Delete the specific history item
        result = await history_collection.delete_one({
            "_id": ObjectId(history_id),
            "user_id": current_user.id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="History item not found"
            )
        
        logger.info(f"History item {history_id} deleted by user {current_user.email}")
        return {"message": "History item deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history item: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error deleting history item"
        )

@router.delete("/history")
async def clear_all_history(
    current_user: Optional[UserInDB] = Depends(get_optional_current_user)
):
    """Clear all conversion history for the user"""
    try:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to clear history"
            )
        
        history_collection = get_history_collection()
        
        # Delete all history items for the user
        result = await history_collection.delete_many({"user_id": current_user.id})
        
        logger.info(f"Cleared {result.deleted_count} history items for user {current_user.email}")
        return {
            "message": f"Successfully cleared {result.deleted_count} history items"
        }
        
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error clearing history"
        )

# Helper function to save conversion history
async def save_conversion_history(
    user_id: Optional[ObjectId],
    filename: str,
    original_filename: str,
    file_size: int,
    conversion_type: str,
    extracted_data: Optional[dict] = None,
    final_output: Optional[dict] = None,
    status: str = "success",
    error_message: Optional[str] = None,
    processing_time: Optional[float] = None
):
    """Save conversion history to database"""
    try:
        history_collection = get_history_collection()
        
        history_item = {
            "user_id": user_id,  # Can be None for guest users
            "filename": filename,
            "original_filename": original_filename,
            "file_size": file_size,
            "conversion_type": conversion_type,
            "extracted_data": extracted_data,
            "final_output": final_output,
            "status": status,
            "error_message": error_message,
            "created_at": datetime.utcnow(),
            "processing_time": processing_time
        }
        
        result = await history_collection.insert_one(history_item)
        logger.info(f"Saved conversion history: {result.inserted_id}")
        return result.inserted_id
        
    except Exception as e:
        logger.error(f"Error saving conversion history: {str(e)}")
        # Don't raise exception here as this shouldn't break the conversion process
        return None 
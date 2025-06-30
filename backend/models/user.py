from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Any, Annotated
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, handler=None):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema, handler):
        field_schema.update(type="string")
        return field_schema

# User Registration Schema
class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)

# User Login Schema
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# User Response Schema (without password)
class UserResponse(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    email: EmailStr
    created_at: datetime
    is_active: bool = True
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

# User Update Schema
class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None

# User in Database
class UserInDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    name: str
    email: EmailStr
    hashed_password: str
    created_at: datetime
    is_active: bool = True
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

# Token Schema
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Conversion History Schema
class ConversionHistory(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: Optional[str] = None  # None for guest users
    filename: str
    original_filename: str
    file_size: int
    conversion_type: str  # 'pdf_to_excel', 'pdf_to_csv', etc.
    extracted_data: Optional[dict] = None
    final_output: Optional[dict] = None
    status: str  # 'success', 'failed', 'processing'
    error_message: Optional[str] = None
    created_at: datetime
    processing_time: Optional[float] = None  # in seconds
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

class ConversionHistoryResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_size: int
    conversion_type: str
    status: str
    created_at: datetime
    processing_time: Optional[float] = None
    
    model_config = {
        "json_encoders": {ObjectId: str}
    } 
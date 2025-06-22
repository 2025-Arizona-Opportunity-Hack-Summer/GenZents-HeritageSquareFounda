from pydantic import BaseModel, ConfigDict
from typing import Optional


class User(BaseModel):
    """Simple user model to avoid Supabase import issues"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    email: Optional[str] = None


class AuthResponse(BaseModel):
    """Authentication response model"""
    user: User
    access_token: str
    token_type: str = "bearer"

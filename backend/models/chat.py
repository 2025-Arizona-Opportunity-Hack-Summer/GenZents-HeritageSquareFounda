from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatCreate(BaseModel):
    title: str = "New Chat"
    model: Optional[str] = "gemini-1.5-flash"
    system_prompt: Optional[str] = None
    tags: Optional[List[str]] = None


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    tags: Optional[List[str]] = None


class ChatResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    user_id: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    tags: Optional[List[str]] = []
    bookmarked: bool = False
    archived: bool = False
    shared: bool = False
    version: int = 1

    class Config:
        from_attributes = True

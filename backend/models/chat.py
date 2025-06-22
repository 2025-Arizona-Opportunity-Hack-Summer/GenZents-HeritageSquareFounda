from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatCreate(BaseModel):
    title: str = Field(default="New Chat", min_length=1, max_length=100)
    model: Optional[str] = Field(default="gemini-1.5-flash", pattern=r"^(gemini-1\.5-flash|gemini-1\.5-pro)$")
    system_prompt: Optional[str] = Field(None, max_length=2000)
    tags: Optional[List[str]] = Field(None)

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        return v.strip() if v else "New Chat"

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if v:
            return [tag.strip()[:50] for tag in v if tag.strip()][:10]
        return v


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    tags: Optional[List[str]] = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
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

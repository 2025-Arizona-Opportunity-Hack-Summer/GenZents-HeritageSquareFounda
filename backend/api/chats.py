from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import List, Optional
from storage.database import supabase, get_current_user, get_user_supabase_client
from models.chat import ChatCreate, ChatResponse, ChatUpdate
from models.user import User
from utils.logger import logger
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/chats", tags=["chats"])
security = HTTPBearer()


def get_authenticated_supabase(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get Supabase client authenticated with user's token"""
    try:
        return get_user_supabase_client(credentials.credentials)
    except Exception as e:
        logger.error(f"Failed to create authenticated Supabase client: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate with database"
        )


@router.get("/", response_model=List[ChatResponse])
async def get_chats(
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase),
    archived: Optional[bool] = None,
    bookmarked: Optional[bool] = None
):
    """Get all chats for the current user"""
    try:
        query = user_supabase.table("chats").select(
            "*").eq("user_id", current_user.id)

        if archived is not None:
            query = query.eq("archived", archived)
        if bookmarked is not None:
            query = query.eq("bookmarked", bookmarked)

        response = query.order("updated_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=ChatResponse)
async def create_chat(
    chat: ChatCreate,
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase)
):
    """Create a new chat"""
    try:
        logger.info(f"Creating chat for user {current_user.id} with title '{chat.title}'")
        
        chat_data = {
            "id": str(uuid.uuid4()),
            "title": chat.title or "New Chat",
            "user_id": current_user.id,
            "model": chat.model or "gemini-1.5-flash",
            "system_prompt": chat.system_prompt,
            "tags": chat.tags or [],
            "bookmarked": False,
            "archived": False,
            "shared": False,
            "version": 1,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Inserting chat data: {chat_data}")
        response = user_supabase.table("chats").insert(chat_data).execute()
        
        if not response.data:
            logger.error("No data returned from chat creation")
            raise HTTPException(status_code=500, detail="Chat creation failed - no data returned")
            
        logger.info(f"Chat created successfully: {response.data[0]['id']}")
        return response.data[0]
        
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error creating chat: {error_detail}")
        
        # Provide more specific error messages
        if "row-level security policy" in error_detail.lower():
            raise HTTPException(
                status_code=403, 
                detail="Permission denied: Unable to create chat. Please contact administrator."
            )
        elif "authentication" in error_detail.lower():
            raise HTTPException(
                status_code=401, 
                detail="Authentication failed. Please log in again."
            )
        elif "duplicate key" in error_detail.lower():
            raise HTTPException(
                status_code=409, 
                detail="Chat ID conflict. Please try again."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Chat creation failed: {error_detail}")


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase)
):
    """Get a specific chat"""
    try:
        response = user_supabase.table("chats").select(
            "*").eq("id", chat_id).eq("user_id", current_user.id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Chat not found")

        return response.data[0]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: str,
    chat: ChatUpdate,
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase)
):
    """Update a chat"""
    try:
        update_data = {}
        if chat.title is not None:
            update_data["title"] = chat.title
        if chat.model is not None:
            update_data["model"] = chat.model
        if chat.system_prompt is not None:
            update_data["system_prompt"] = chat.system_prompt
        if chat.tags is not None:
            update_data["tags"] = chat.tags

        update_data["updated_at"] = datetime.utcnow().isoformat()

        response = user_supabase.table("chats").update(update_data).eq(
            "id", chat_id).eq("user_id", current_user.id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Chat not found")

        return response.data[0]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase)
):
    """Delete a chat"""
    try:
        # First delete all messages in the chat
        user_supabase.table("messages").delete().eq("chat_id", chat_id).execute()

        # Then delete the chat
        response = user_supabase.table("chats").delete().eq(
            "id", chat_id).eq("user_id", current_user.id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Chat not found")

        return {"message": "Chat deleted successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

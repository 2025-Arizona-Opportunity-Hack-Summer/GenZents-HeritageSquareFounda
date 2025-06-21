from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import List, Optional
from storage.database import supabase, get_current_user, get_user_supabase_client
from models.message import MessageCreate, MessageResponse, MessageUpdate
from models.user import User
from utils.logger import logger
import uuid
import json
from datetime import datetime
from services.generative_ai import generate_text, generate_text_stream

router = APIRouter(prefix="/api/messages", tags=["messages"])
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


@router.get("/chat/{chat_id}", response_model=List[MessageResponse])
async def get_messages(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase),
    limit: Optional[int] = 50,
    offset: Optional[int] = 0
):
    """Get all messages for a specific chat"""
    try:
        # First verify user has access to this chat
        chat_response = user_supabase.table("chats").select("id").eq(
            "id", chat_id).eq("user_id", current_user.id).execute()

        if not chat_response.data:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Get messages
        response = user_supabase.table("messages").select("*").eq("chat_id", chat_id).order(
            "created_at", desc=False).range(offset, offset + limit - 1).execute()

        return response.data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/{chat_id}", response_model=MessageResponse)
async def create_message(
    chat_id: str,
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase)
):
    """Create a new message in a chat and get a response from the AI"""
    try:
        logger.info(f"Creating message for chat {chat_id} by user {current_user.id}")
        
        # 1. Verify user has access to this chat
        chat_response = user_supabase.table("chats").select("id, system_prompt").eq(
            "id", chat_id).eq("user_id", current_user.id).execute()

        if not chat_response.data:
            logger.warning(f"Chat {chat_id} not found for user {current_user.id}")
            raise HTTPException(status_code=404, detail="Chat not found")

        if message.role != 'user':
            raise HTTPException(
                status_code=400, detail="Only user messages can be created through this endpoint.")

        # 2. Get chat history BEFORE adding the new message
        history_response = user_supabase.table("messages").select(
            "role, content").eq("chat_id", chat_id).order("created_at").execute()
        chat_history = history_response.data
        
        # 3. Save user's message
        user_message_data = {
            "id": str(uuid.uuid4()),
            "chat_id": chat_id,
            "role": "user",
            "content": message.content,
            "created_at": datetime.utcnow().isoformat()
        }
        user_response = user_supabase.table("messages").insert(user_message_data).execute()
        logger.info(f"User message saved: {user_message_data['id']}")

        # 4. Generate AI response
        system_prompt = chat_response.data[0].get('system_prompt')

        ai_response_text = generate_text(
            prompt=message.content,
            history=chat_history,
            system_prompt=system_prompt
        )

        # 5. Save AI's response
        ai_message_data = {
            "id": str(uuid.uuid4()),
            "chat_id": chat_id,
            "role": "assistant",
            "content": ai_response_text,
            "created_at": datetime.utcnow().isoformat()
        }
        response = user_supabase.table("messages").insert(ai_message_data).execute()

        # 6. Update chat's updated_at timestamp
        user_supabase.table("chats").update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", chat_id).execute()

        logger.info(f"AI message saved: {ai_message_data['id']}")
        return response.data[0]

    except HTTPException:
        raise
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error creating message in chat {chat_id}: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Message creation failed: {error_detail}")


@router.post("/chat/{chat_id}/stream")
async def create_message_stream(
    chat_id: str,
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    user_supabase = Depends(get_authenticated_supabase)
):
    """Create a new message and stream AI response"""
    try:
        logger.info(f"Creating streaming message for chat {chat_id} by user {current_user.id}")
        
        # Verify user has access to this chat
        chat_response = user_supabase.table("chats").select("id, system_prompt").eq(
            "id", chat_id).eq("user_id", current_user.id).execute()

        if not chat_response.data:
            logger.warning(f"Chat {chat_id} not found for user {current_user.id}")
            raise HTTPException(status_code=404, detail="Chat not found")

        if message.role != 'user':
            raise HTTPException(
                status_code=400, detail="Only user messages can be created through this endpoint.")

        # Get chat history
        history_response = user_supabase.table("messages").select(
            "role, content").eq("chat_id", chat_id).order("created_at").execute()
        chat_history = history_response.data
        
        # Save user's message
        user_message_data = {
            "id": str(uuid.uuid4()),
            "chat_id": chat_id,
            "role": "user",
            "content": message.content,
            "created_at": datetime.utcnow().isoformat()
        }
        user_supabase.table("messages").insert(user_message_data).execute()
        logger.info(f"User message saved for streaming: {user_message_data['id']}")

        async def generate_response():
            try:
                system_prompt = chat_response.data[0].get('system_prompt')

                # Stream the AI response
                full_response = ""
                async for chunk in generate_text_stream(
                    prompt=message.content,
                    history=chat_history,
                    system_prompt=system_prompt
                ):
                    full_response += chunk

                    # Send chunk to client
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

                # Save the complete AI response to database
                ai_message_data = {
                    "id": str(uuid.uuid4()),
                    "chat_id": chat_id,
                    "role": "assistant",
                    "content": full_response,
                    "created_at": datetime.utcnow().isoformat()
                }
                ai_response = user_supabase.table("messages").insert(
                    ai_message_data).execute()

                # Update chat timestamp
                user_supabase.table("chats").update({
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", chat_id).execute()

                logger.info(f"AI streaming message saved: {ai_message_data['id']}")
                # Send completion signal with the saved message
                yield f"data: {json.dumps({'type': 'complete', 'message': ai_response.data[0]})}\n\n"

            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error creating streaming message in chat {chat_id}: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Streaming message creation failed: {error_detail}")


@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific message"""
    try:
        # Get message and verify user has access to the chat
        response = supabase.table("messages").select(
            "*, chats!inner(user_id)").eq("id", message_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Message not found")

        message = response.data[0]

        # Check if user owns the chat
        if message["chats"]["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Remove the nested chats data
        del message["chats"]
        return message
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{message_id}", response_model=MessageResponse)
async def update_message(
    message_id: str,
    message: MessageUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a message"""
    try:
        # Get message and verify user has access to the chat
        response = supabase.table("messages").select(
            "*, chats!inner(user_id)").eq("id", message_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Message not found")

        existing_message = response.data[0]

        # Check if user owns the chat
        if existing_message["chats"]["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Update message
        update_data = {"updated_at": datetime.utcnow().isoformat()}
        if message.content is not None:
            update_data["content"] = message.content
        if message.metadata is not None:
            update_data["metadata"] = message.metadata

        response = supabase.table("messages").update(
            update_data).eq("id", message_id).execute()

        return response.data[0]
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{message_id}")
async def delete_message(
    message_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a message"""
    try:
        # Get message and verify user has access to the chat
        response = supabase.table("messages").select(
            "*, chats!inner(user_id)").eq("id", message_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Message not found")

        message = response.data[0]

        # Check if user owns the chat
        if message["chats"]["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Delete message
        supabase.table("messages").delete().eq("id", message_id).execute()

        return {"message": "Message deleted successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

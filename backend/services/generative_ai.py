import google.generativeai as genai
from config import settings
from typing import List, Dict, Optional, AsyncIterator
from utils.logger import logger
import asyncio

if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
else:
    logger.warning("Google API key not configured")


def format_chat_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format chat history for Google Generative AI"""
    formatted_history = []

    for message in history:
        role = message.get('role', 'user')
        content = message.get('content', '')

        if role == 'user':
            formatted_history.append({
                'role': 'user',
                'parts': [content]
            })
        elif role == 'assistant':
            formatted_history.append({
                'role': 'model',
                'parts': [content]
            })

    return formatted_history


def generate_text(prompt: str, history: List[Dict[str, str]] = [], system_prompt: Optional[str] = None) -> str:
    """
    Generates text using the Google Generative AI API with chat context.

    Args:
        prompt (str): The user's prompt.
        history (List[Dict[str, str]]): The chat history.
        system_prompt (Optional[str]): System prompt for the model.

    Returns:
        str: The generated text.
    """
    if not settings.GEMINI_API_KEY:
        return "AI service is not configured. Please check your API key."

    # Input validation
    if not prompt or len(prompt.strip()) == 0:
        return "Please provide a valid message."
    
    if len(prompt) > 8000:
        return "Message too long. Please limit your message to 8000 characters."

    try:
        # Configure model with system instruction if provided
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_prompt if system_prompt else "You are a helpful AI assistant."
        )

        # Format history for the API
        formatted_history = format_chat_history(history)

        # Start chat with history
        chat = model.start_chat(history=formatted_history)

        # Send the new message
        response = chat.send_message(prompt)

        return response.text

    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."


async def generate_text_stream(prompt: str, history: List[Dict[str, str]] = [], system_prompt: Optional[str] = None) -> AsyncIterator[str]:
    """
    Generates streaming text using the Google Generative AI API with chat context.

    Args:
        prompt (str): The user's prompt.
        history (List[Dict[str, str]]): The chat history.
        system_prompt (Optional[str]): System prompt for the model.

    Yields:
        str: Chunks of generated text.
    """
    if not settings.GEMINI_API_KEY:
        yield "AI service is not configured. Please check your API key."
        return

    try:
        # Configure model with system instruction if provided
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_prompt if system_prompt else "You are a helpful AI assistant."
        )

        # Format history for the API
        formatted_history = format_chat_history(history)

        # Start chat with history
        chat = model.start_chat(history=formatted_history)

        # Generate streaming response
        response = chat.send_message(prompt, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"Error generating streaming text: {e}")
        yield "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."

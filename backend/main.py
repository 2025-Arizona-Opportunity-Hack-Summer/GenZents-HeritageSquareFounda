from api import chats, messages
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import settings
from utils.logger import logger

# import sys
# from pathlib import Path

# # Add the parent directory to the path so we can import our modules
# sys.path.append(str(Path(__file__).parent))

# Import routers

# Initialize FastAPI app
app = FastAPI(
    title="Archyx AI API",
    description="Backend API for the Archyx AI chat application",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL, 
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://your-frontend-domain.com",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Add preflight handler for CORS
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return {"detail": "OK"}

# Health check endpoint


@app.get("/")
async def root():
    logger.info("Health check endpoint accessed")
    return {"message": "Archyx AI API is running"}

# Include routers
app.include_router(chats.router)
app.include_router(messages.router)

if __name__ == "__main__":
    logger.info(f"Starting Archyx AI API on {settings.HOST}:{settings.PORT}")
    uvicorn.run("main:app", host=settings.HOST,
                port=settings.PORT, reload=settings.DEBUG)

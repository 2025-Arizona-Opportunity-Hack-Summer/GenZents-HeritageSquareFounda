from supabase import create_client, Client, ClientOptions
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from models.user import User
from config import settings
from utils.logger import logger

# Initialize clients
supabase: Client = None
if settings.is_configured:
    try:
        supabase = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        logger.error(
            "Please check your SUPABASE_URL and SUPABASE_ANON_KEY in .env file")
        supabase = None
else:
    logger.warning(
        "Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file")

security = HTTPBearer()

# Function to get authenticated supabase client for user requests  
def get_user_supabase_client(token: str) -> Client:
    """Create a Supabase client with user authentication token for RLS"""
    try:
        # Validate token format
        if not token or len(token) < 10:
            raise ValueError("Invalid token format")
            
        # Create options with proper headers
        options = ClientOptions(
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": settings.SUPABASE_ANON_KEY
            }
        )
        
        # Create a new client with user authentication
        user_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_ANON_KEY,
            options=options
        )
        
        return user_client
    except Exception as e:
        logger.error(f"Failed to create authenticated Supabase client: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service not configured"
        )

    token = credentials.credentials
    
    # Basic token format validation
    if not token or len(token) < 10 or not token.replace('-', '').replace('_', '').replace('.', '').isalnum():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format"
        )
    
    try:
        # Verify JWT token with Supabase
        response = supabase.auth.get_user(token)
        if response.user:
            return User(id=response.user.id, email=response.user.email)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except Exception as e:
        logger.error("Authentication failed: Invalid credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

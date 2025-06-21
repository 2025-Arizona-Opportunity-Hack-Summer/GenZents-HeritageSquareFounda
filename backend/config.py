import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

    # CORS Settings
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # Google Generative AI Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Google Drive Configuration
    GOOGLE_CREDENTIALS_PATH: str = os.getenv(
        "GOOGLE_CREDENTIALS_PATH", "credentials.json")
    GOOGLE_TOKEN_PATH: str = os.getenv("GOOGLE_TOKEN_PATH", "token.json")

    @classmethod
    def validate(cls, raise_on_missing: bool = True):
        """Validate required environment variables"""
        from utils.logger import logger

        required_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY", "GEMINI_API_KEY"]
        missing_vars = [var for var in required_vars if not getattr(cls, var)]

        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.warning(error_msg)
            logger.info(
                "Please create a .env file with your Supabase credentials:")
            logger.info("SUPABASE_URL=your_supabase_project_url")
            logger.info("SUPABASE_ANON_KEY=your_supabase_anon_key")
            logger.info("GEMINI_API_KEY=your_google_api_key")

            if raise_on_missing:
                raise ValueError(error_msg)
            return False

        return True

    @property
    def is_configured(self) -> bool:
        """Check if minimum configuration is present"""
        return bool(self.SUPABASE_URL and self.SUPABASE_ANON_KEY)

    @property
    def is_google_drive_configured(self) -> bool:
        """Check if Google Drive configuration is present"""
        import os
        return (os.path.exists(self.GOOGLE_CREDENTIALS_PATH) or
                bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')))


settings = Settings()

# Only validate if we're not in development mode
# This allows the app to start even without env vars for testing
if not settings.DEBUG:
    settings.validate()

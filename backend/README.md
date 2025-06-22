This is the backend API server supporting the frontend application. It handles authentication, AI agent interactions, file management, and script execution.

## Features
- Secure API endpoints with Supabase authentication
- Chat management (create, read, update, delete)
- Message handling and AI response generation
- Analytics and usage tracking
- Integration with OpenAI and Anthropic AI models
- RESTful API design with automatic documentation

## Tech Stack
- **FastAPI**: Modern, fast web framework for building APIs
- **Supabase**: Backend-as-a-Service for database and authentication
- **Pydantic**: Data validation using Python type annotations
- **OpenAI**: GPT models integration
- **Anthropic**: Claude models integration

## Setup

### Prerequisites
- Python 3.8 or higher
- Supabase account and project
- OpenAI API key (optional)
- Anthropic API key (optional)

### Installation

1. Clone the repository and navigate to the backend directory
```bash
cd backend
```

2. Run the startup script (automatically creates venv and installs dependencies):

**For Windows:**
```bash
start.bat
```

**For Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

3. Configure environment variables in `.env`:
```env
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

4. Start the development server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

Once the server is running, you can access:
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Authentication
All endpoints require Bearer token authentication via Supabase JWT.

### Chats
- `GET /api/chats/` - Get all chats for the user
- `POST /api/chats/` - Create a new chat
- `GET /api/chats/{chat_id}` - Get a specific chat
- `PUT /api/chats/{chat_id}` - Update a chat
- `DELETE /api/chats/{chat_id}` - Delete a chat
- `POST /api/chats/{chat_id}/archive` - Archive/unarchive a chat
- `POST /api/chats/{chat_id}/bookmark` - Bookmark/unbookmark a chat

### Messages
- `GET /api/messages/chat/{chat_id}` - Get messages for a chat
- `POST /api/messages/chat/{chat_id}` - Create a new message
- `GET /api/messages/{message_id}` - Get a specific message
- `PUT /api/messages/{message_id}` - Update a message
- `DELETE /api/messages/{message_id}` - Delete a message

### AI
- `POST /api/ai/generate` - Generate AI response
- `POST /api/ai/chat/{chat_id}/generate` - Generate AI response for a chat
- `GET /api/ai/models` - Get available AI models

### Analytics
- `GET /api/analytics/usage` - Get usage metrics
- `GET /api/analytics/chats/stats` - Get chat statistics
- `POST /api/analytics/events` - Track analytics events
- `GET /api/analytics/events` - Get analytics events

## Database Schema

The backend uses Supabase with the following main tables:
- `chats` - Chat conversations
- `messages` - Individual messages within chats
- `usage_logs` - AI API usage tracking
- `analytics` - User analytics events
- `ai_models` - Available AI models

## Security
- JWT token authentication via Supabase
- Row Level Security (RLS) policies
- API key protection for AI services
- CORS configuration for frontend integration

## Development

### Project Structure
```
backend/
├── main.py              # FastAPI application entry point
├── routers/             # API route handlers
│   ├── __init__.py
│   ├── chats.py         # Chat management endpoints
│   ├── messages.py      # Message handling endpoints
│   ├── ai.py            # AI integration endpoints
│   └── analytics.py     # Analytics endpoints
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── start.sh            # Linux/Mac startup script
└── start.bat           # Windows startup script
```

### Contributing
1. Follow PEP 8 style guidelines
2. Add type hints for all functions
3. Include docstrings for API endpoints
4. Test endpoints using the built-in docs

## License
MIT
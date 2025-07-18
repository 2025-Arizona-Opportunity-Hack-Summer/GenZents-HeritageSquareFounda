from typing import Dict, Any, Optional, Callable
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from datetime import datetime
import uuid

from config import settings
from scripts.google_drive import GoogleDriveService
from utils.logger import logger
from services.user_security import get_security_service
from storage.database import get_user_supabase_client


class GoogleDriveAgent:
    def __init__(self, user_id: Optional[str]
                 = None, user_supabase_client=None):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.drive_service = GoogleDriveService()
        self.model = self._create_model()
        self.chat = None
        self.user_id = user_id or "anonymous"
        self.user_supabase = user_supabase_client
        self.security_service = get_security_service(
            user_supabase_client) if user_supabase_client else None

    def _create_model(self) -> genai.GenerativeModel:
        """Create Gemini model with Google Drive function tools"""
        tools = [Tool(function_declarations=[
            FunctionDeclaration(
                name="list_files",
                description="List files in Google Drive. Can filter by folder or search query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "folder_id": {
                            "type": "string",
                            "description": "Optional Google Drive folder ID to list files from"
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional search query to filter files"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 50)"
                        }
                    }
                }
            ),
            FunctionDeclaration(
                name="get_file_info",
                description="Get detailed information about a specific file by its ID",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Google Drive file ID"
                        }
                    },
                    "required": ["file_id"]
                }
            ),
            FunctionDeclaration(
                name="create_folder",
                description="Create a new folder in Google Drive",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the folder to create"
                        },
                        "parent_id": {
                            "type": "string",
                            "description": "Optional parent folder ID"
                        }
                    },
                    "required": ["name"]
                }
            ),
            FunctionDeclaration(
                name="move_file",
                description="Move a file to a different folder",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "ID of the file to move"
                        },
                        "new_parent_id": {
                            "type": "string",
                            "description": "ID of the destination folder"
                        }
                    },
                    "required": ["file_id", "new_parent_id"]
                }
            ),
            FunctionDeclaration(
                name="rename_file",
                description="Rename a file or folder",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "ID of the file to rename"
                        },
                        "new_name": {
                            "type": "string",
                            "description": "New name for the file"
                        }
                    },
                    "required": ["file_id", "new_name"]
                }
            ),
            FunctionDeclaration(
                name="delete_file",
                description="Delete a file (move to trash). Use with caution!",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "ID of the file to delete"
                        }
                    },
                    "required": ["file_id"]
                }
            ),
            FunctionDeclaration(
                name="search_files",
                description="Search for files by name or content",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 50)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            FunctionDeclaration(
                name="get_storage_info",
                description="Get Google Drive storage information",
                parameters={"type": "object", "properties": {}}
            ),
            FunctionDeclaration(
                name="organize_by_type",
                description="Organize files by type within a folder",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_folder_id": {
                            "type": "string",
                            "description": "ID of the folder to organize"
                        }
                    },
                    "required": ["source_folder_id"]
                }
            ),
            FunctionDeclaration(
                name="get_folder_structure",
                description="Get hierarchical folder structure",
                parameters={
                    "type": "object",
                    "properties": {
                        "folder_id": {
                            "type": "string",
                            "description": "Starting folder ID (optional, defaults to root)"
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum depth to traverse (default: 3)"
                        }
                    }
                }
            )
        ])]

        generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        system_instruction = """You are a Google Drive management assistant. You can help users organize, search, and manage their Google Drive files safely.

IMPORTANT SAFETY RULES:
1. Always confirm before deleting files - ask the user to confirm deletion operations
2. When moving or renaming files, explain what you're about to do
3. For bulk operations, process a few files at a time and ask for confirmation
4. If a user asks for destructive operations on many files, ask for explicit confirmation
5. Provide helpful summaries of what was accomplished

You have access to the following Google Drive functions:
- List, search, and get file information
- Create folders and organize files
- Move, rename, and delete files
- Share files with users
- Get storage information
- Organize files by type

Be helpful, safe, and provide clear explanations of what you're doing."""

        return genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=generation_config,
            tools=tools,
            system_instruction=system_instruction
        )

    def _get_function_mapping(self) -> Dict[str, Callable]:
        """Map function names to actual drive service methods"""
        return {
            "list_files": self.drive_service.list_files,
            "get_file_info": self.drive_service.get_file_info,
            "create_folder": self.drive_service.create_folder,
            "move_file": self.drive_service.move_file,
            "rename_file": self.drive_service.rename_file,
            "delete_file": self.drive_service.delete_file,
            "search_files": self.drive_service.search_files,
            "get_storage_info": self.drive_service.get_storage_info,
            "organize_by_type": self.drive_service.organize_by_type,
            "get_folder_structure": self.drive_service.get_folder_structure
        }

    def start_chat(self) -> None:
        """Start a new chat session"""
        self.chat = self.model.start_chat()
        logger.info(
            f"""Started new Google Drive agent chat session for user {self.user_id}""")

    def handle_regular_query(self, query:str):
        try:
            logger.info(f"Processing query for user {self.user_id}")

            from services.rag import RAG
            my_rag = RAG(self.drive)
            context = my_rag.retrieve(query)
            # message = my_rag.generate_message(context=context,query=query)
            # print(message)
            
            prompt = f"""Question: {query}

                Retrieved Context:
                {context}

                Please provide a clear, relevant answer based on the above context."""

            if not self.chat:
                self.start_chat()

            response = self.chat.send_message(prompt)
            return response.text if response.text else "No relevant information found."

        except Exception as e:
            logger.error(f"Error in process_regular_query: {str(e)}")
            raise ValueError(f"Query processing failed: {str(e)}")
    async def process_message(self, message: str) -> str:
        """Process user message and execute any necessary function calls"""
        if not self.chat:
            self.start_chat()

        try:
            logger.info(
                f"""Processing user message for user {self.user_id}: {message}""")
            response = self.chat.send_message(message)

            # Handle function calls
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        result = await self._safe_execute_function(function_call)

                        # Send function result back to the model
                        function_response = genai.protos.FunctionResponse(
                            name=function_call.name,
                            response={"result": result}
                        )
                        response = self.chat.send_message(function_response)

            return response.text if response.text else "I've completed the requested action."

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error: {str(e)}. Please try again or rephrase your request."

    async def _check_user_permissions(
            self, operation: str, resource_path: str = "") -> bool:
        """Check if user has permission to perform the operation"""
        if not self.security_service or not self.user_id:
            return False

        try:
            # Get user constraints to check permissions
            constraints = await self.security_service.get_user_constraints(self.user_id)

            # Permission mapping for different operations
            permission_map = {
                "list_files": "file_organization",
                "get_file_info": "file_organization",
                "create_folder": "file_organization",
                "move_file": "file_organization",
                "rename_file": "file_organization",
                "delete_file": "file_organization",
                "search_files": "file_organization",
                "organize_by_type": "file_organization",
                "get_folder_structure": "file_organization"
            }

            required_permission = permission_map.get(
                operation, "file_organization")

            # Admin can do everything
            if constraints.is_admin:
                return True

            # Check if user has required permission
            return required_permission in constraints.permissions

        except Exception as e:
            logger.error(
                f"""Error checking permissions for user {self.user_id}: {e}""")
            return False
    

    async def _track_change(self, change_type: str, resource_path: str,
                            old_path: str = None, new_path: str = None,
                            metadata: Dict[str, Any] = None) -> None:
        """Track file/folder changes using existing changes table structure only"""
        if not self.user_supabase or not self.user_id:
            return

        try:
            # First create or get version entry
            version_id = await self._get_or_create_version(resource_path, f"{change_type.title()} operation")

            # Insert using only the existing table structure from database.ts
            change_data = {
                "version_id": version_id,
                "type": self._map_change_type(change_type),
                "original_path": old_path or resource_path,
                "new_path": new_path,
                "description": f"{change_type} operation on {resource_path}",
                "user_id": self.user_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            result = self.user_supabase.table(
                "changes").insert(change_data).execute()
            logger.info(f"Tracked change: {change_type} on {resource_path}")

        except Exception as e:
            logger.error(f"Error tracking change: {e}")

    def _map_change_type(self, change_type: str) -> str:
        """Map our change types to existing table's type values from database.ts"""
        mapping = {
            "create": "added",
            "update": "modified",
            "delete": "deleted",
            "move": "modified",
            "rename": "modified",
            "organize": "modified"
        }
        return mapping.get(change_type, "modified")

    async def _get_or_create_version(
            self, resource_path: str, description: str) -> str:
        """Get existing version or create new one using existing table structure from database.ts"""
        try:
            # Check for existing current version for this user
            existing_response = self.user_supabase.table("versions").select("id").eq(
                "user_id", self.user_id
            ).eq("status", "current").limit(1).execute()

            if existing_response.data:
                return existing_response.data[0]["id"]

            # Create new version using only existing structure from database.ts
            version_data = {
                "version": f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "title": f"Drive Changes",
                "description": description,
                "user_id": self.user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "current",
                "data": {"resource_path": resource_path, "change_type": description}
            }

            version_response = self.user_supabase.table(
                "versions").insert(version_data).execute()
            return version_response.data[0]["id"]

        except Exception as e:
            logger.error(f"Error creating version entry: {e}")
            # Return a placeholder if version creation fails
            return str(uuid.uuid4())

    async def _safe_execute_function(self, function_call) -> Any:
        """Execute a Google Drive function with permission checking and change tracking"""
        function_name = function_call.name
        function_args = dict(function_call.args) if function_call.args else {}

        # Check permissions first
        resource_path = function_args.get(
            "file_id", function_args.get(
                "folder_id", ""))
        has_permission = await self._check_user_permissions(function_name, resource_path)

        if not has_permission:
            error_msg = f"Permission denied for operation: {function_name}"
            logger.warning(
                f"""User {self.user_id} denied permission for {function_name}""")
            return {"error": error_msg}

        # Execute the original function
        try:
            function_mapping = self._get_function_mapping()
            func = function_mapping[function_name]
            result = func(**function_args)

            # Track changes for modification operations
            if function_name in ["move_file", "rename_file",
                                 "delete_file", "create_folder", "organize_by_type"]:
                await self._track_file_operation(function_name, function_args, result)

            return result

        except Exception as e:
            error_msg = f"Error executing {function_name}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def _track_file_operation(
            self, operation: str, args: Dict[str, Any], result: Any) -> None:
        """Track specific file operations using existing table structure"""
        try:
            if operation == "move_file":
                await self._track_change("move", args.get("file_id", ""),
                                         old_path=args.get("file_id"),
                                         new_path=args.get("new_parent_id"))

            elif operation == "rename_file":
                await self._track_change("rename", args.get("file_id", ""),
                                         old_path=args.get("file_id"),
                                         new_path=args.get("new_name"))

            elif operation == "delete_file":
                await self._track_change("delete", args.get("file_id", ""))

            elif operation == "create_folder":
                await self._track_change("create", args.get("name", ""))

            elif operation == "organize_by_type":
                await self._track_change("organize", args.get("source_folder_id", ""))

        except Exception as e:
            logger.error(f"Error tracking file operation {operation}: {e}")

    def reset_chat(self) -> None:
        """Reset the chat session"""
        self.chat = None
        logger.info("Reset Google Drive agent chat session")


def create_drive_agent(user_id: str = None,
                       user_supabase_client=None) -> GoogleDriveAgent:
    """Factory function to create a Google Drive agent with user context"""
    return GoogleDriveAgent(
        user_id=user_id, user_supabase_client=user_supabase_client)

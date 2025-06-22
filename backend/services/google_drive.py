import os
import io
from typing import Dict, List, Optional, Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from config import settings
from utils.logger import logger


class GoogleDriveService:
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.metadata'
    ]

    def __init__(self, credentials_path: Optional[str] = None, token_path: Optional[str] = None):
        self.credentials_path = credentials_path or settings.GOOGLE_CREDENTIALS_PATH
        self.token_path = token_path or settings.GOOGLE_TOKEN_PATH
        self.service = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Google Drive API"""
        creds = None

        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(
                    self.token_path, self.SCOPES)
            except Exception as e:
                logger.warning(f"Failed to load existing credentials: {e}")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("Credentials refreshed successfully")
                except RefreshError as e:
                    logger.error(f"Failed to refresh credentials: {e}")
                    creds = None

            if not creds:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_path}")

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
                logger.info("Credentials saved successfully")

        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive service initialized")

    def get_file_info(self, file_id: str, fields: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a file"""
        try:
            default_fields = 'id,name,mimeType,size,createdTime,modifiedTime,parents,webViewLink,webContentLink,owners,permissions'
            fields = fields or default_fields

            file_info = self.service.files().get(fileId=file_id, fields=fields).execute()
            return self._format_file_info(file_info)
        except HttpError as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            raise

    def list_files(self, folder_id: Optional[str] = None, query: Optional[str] = None,
                   max_results: int = 100, order_by: str = 'modifiedTime desc') -> List[Dict[str, Any]]:
        """List files in a folder or matching a query"""
        try:
            q = []
            if folder_id:
                q.append(f"'{folder_id}' in parents")
            if query:
                q.append(query)
            q.append("trashed=false")

            query_string = " and ".join(q)

            results = self.service.files().list(
                q=query_string,
                pageSize=max_results,
                orderBy=order_by,
                fields="nextPageToken, files(id,name,mimeType,size,createdTime,modifiedTime,parents,webViewLink)"
            ).execute()

            files = results.get('files', [])
            return [self._format_file_info(file) for file in files]
        except HttpError as e:
            logger.error(f"Failed to list files: {e}")
            raise

    def create_folder(self, name: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new folder"""
        try:
            folder_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }

            if parent_id:
                folder_metadata['parents'] = [parent_id]

            folder = self.service.files().create(
                body=folder_metadata,
                fields='id,name,parents,webViewLink,createdTime'
            ).execute()

            logger.info(f"Created folder: {name} (ID: {folder['id']})")
            return self._format_file_info(folder)
        except HttpError as e:
            logger.error(f"Failed to create folder {name}: {e}")
            raise

    def upload_file(self, file_content: bytes, filename: str, parent_id: Optional[str] = None,
                    mime_type: Optional[str] = None) -> Dict[str, Any]:
        """Upload a file to Google Drive"""
        try:
            file_metadata = {'name': filename}
            if parent_id:
                file_metadata['parents'] = [parent_id]

            media = MediaIoBaseUpload(
                io.BytesIO(file_content),
                mimetype=mime_type or 'application/octet-stream',
                resumable=True
            )

            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,size,webViewLink,createdTime'
            ).execute()

            logger.info(f"Uploaded file: {filename} (ID: {file['id']})")
            return self._format_file_info(file)
        except HttpError as e:
            logger.error(f"Failed to upload file {filename}: {e}")
            raise

    def download_file(self, file_id: str) -> bytes:
        """Download file content"""
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            file_content.seek(0)
            content = file_content.read()
            logger.info(
                f"Downloaded file (ID: {file_id}), size: {len(content)} bytes")
            return content
        except HttpError as e:
            logger.error(f"Failed to download file {file_id}: {e}")
            raise

    def delete_file(self, file_id: str) -> bool:
        """Delete a file (move to trash)"""
        try:
            self.service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted file (ID: {file_id})")
            return True
        except HttpError as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False

    def move_file(self, file_id: str, new_parent_id: str, old_parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Move a file to a different folder"""
        try:
            if not old_parent_id:
                file_info = self.service.files().get(fileId=file_id, fields='parents').execute()
                old_parent_id = file_info.get('parents', [None])[0]

            file = self.service.files().update(
                fileId=file_id,
                addParents=new_parent_id,
                removeParents=old_parent_id,
                fields='id,name,parents'
            ).execute()

            logger.info(
                f"Moved file (ID: {file_id}) to folder (ID: {new_parent_id})")
            return self._format_file_info(file)
        except HttpError as e:
            logger.error(f"Failed to move file {file_id}: {e}")
            raise

    def rename_file(self, file_id: str, new_name: str) -> Dict[str, Any]:
        """Rename a file"""
        try:
            file = self.service.files().update(
                fileId=file_id,
                body={'name': new_name},
                fields='id,name,modifiedTime'
            ).execute()

            logger.info(f"Renamed file (ID: {file_id}) to: {new_name}")
            return self._format_file_info(file)
        except HttpError as e:
            logger.error(f"Failed to rename file {file_id}: {e}")
            raise

    def search_files(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for files by name or content"""
        try:
            search_query = f"name contains '{query}' or fullText contains '{query}'"
            return self.list_files(query=search_query, max_results=max_results)
        except HttpError as e:
            logger.error(f"Failed to search files with query '{query}': {e}")
            raise

    def get_folder_structure(self, folder_id: Optional[str] = None, max_depth: int = 3) -> Dict[str, Any]:
        """Get hierarchical folder structure"""
        try:
            if not folder_id:
                folder_id = 'root'

            folder_info = self.get_file_info(folder_id)
            structure = {
                'info': folder_info,
                'children': []
            }

            if max_depth > 0:
                files = self.list_files(folder_id)
                for file in files:
                    if file['mimeType'] == 'application/vnd.google-apps.folder':
                        child_structure = self.get_folder_structure(
                            file['id'], max_depth - 1
                        )
                        structure['children'].append(child_structure)
                    else:
                        structure['children'].append(
                            {'info': file, 'children': []})

            return structure
        except HttpError as e:
            logger.error(
                f"Failed to get folder structure for {folder_id}: {e}")
            raise

    def get_file_permissions(self, file_id: str) -> List[Dict[str, Any]]:
        """Get file sharing permissions"""
        try:
            permissions = self.service.permissions().list(fileId=file_id).execute()
            return permissions.get('permissions', [])
        except HttpError as e:
            logger.error(f"Failed to get permissions for file {file_id}: {e}")
            raise

    def share_file(self, file_id: str, email: str, role: str = 'reader') -> Dict[str, Any]:
        """Share a file with a user"""
        try:
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }

            result = self.service.permissions().create(
                fileId=file_id,
                body=permission,
                sendNotificationEmail=True
            ).execute()

            logger.info(f"Shared file (ID: {file_id}) with {email} as {role}")
            return result
        except HttpError as e:
            logger.error(f"Failed to share file {file_id} with {email}: {e}")
            raise

    def organize_by_type(self, source_folder_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Organize files by type within a folder"""
        try:
            files = self.list_files(source_folder_id)
            organized = {
                'documents': [],
                'images': [],
                'videos': [],
                'audio': [],
                'spreadsheets': [],
                'presentations': [],
                'pdfs': [],
                'folders': [],
                'others': []
            }

            for file in files:
                mime_type = file.get('mimeType', '')

                if mime_type == 'application/vnd.google-apps.folder':
                    organized['folders'].append(file)
                elif 'document' in mime_type or 'text' in mime_type:
                    organized['documents'].append(file)
                elif 'image' in mime_type:
                    organized['images'].append(file)
                elif 'video' in mime_type:
                    organized['videos'].append(file)
                elif 'audio' in mime_type:
                    organized['audio'].append(file)
                elif 'spreadsheet' in mime_type:
                    organized['spreadsheets'].append(file)
                elif 'presentation' in mime_type:
                    organized['presentations'].append(file)
                elif 'pdf' in mime_type:
                    organized['pdfs'].append(file)
                else:
                    organized['others'].append(file)

            return organized
        except HttpError as e:
            logger.error(
                f"Failed to organize files in folder {source_folder_id}: {e}")
            raise

    def get_storage_info(self) -> Dict[str, Any]:
        """Get Google Drive storage information"""
        try:
            about = self.service.about().get(fields='storageQuota,user').execute()

            storage_quota = about.get('storageQuota', {})
            user_info = about.get('user', {})

            total = int(storage_quota.get('limit', 0))
            used = int(storage_quota.get('usage', 0))
            available = total - used if total > 0 else 0

            return {
                'user': {
                    'email': user_info.get('emailAddress'),
                    'name': user_info.get('displayName')
                },
                'storage': {
                    'total_bytes': total,
                    'used_bytes': used,
                    'available_bytes': available,
                    'total_gb': round(total / (1024**3), 2),
                    'used_gb': round(used / (1024**3), 2),
                    'available_gb': round(available / (1024**3), 2),
                    'usage_percentage': round((used / total * 100), 2) if total > 0 else 0
                }
            }
        except HttpError as e:
            logger.error(f"Failed to get storage info: {e}")
            raise

    def _format_file_info(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format file information for consistent output"""
        formatted = {
            'id': file_info.get('id'),
            'name': file_info.get('name'),
            'mimeType': file_info.get('mimeType'),
            'size': int(file_info.get('size', 0)) if file_info.get('size') else None,
            'createdTime': file_info.get('createdTime'),
            'modifiedTime': file_info.get('modifiedTime'),
            'webViewLink': file_info.get('webViewLink'),
            'webContentLink': file_info.get('webContentLink'),
            'parents': file_info.get('parents', []),
            'isFolder': file_info.get('mimeType') == 'application/vnd.google-apps.folder'
        }

        if file_info.get('owners'):
            formatted['owners'] = [
                {
                    'displayName': owner.get('displayName'),
                    'emailAddress': owner.get('emailAddress')
                }
                for owner in file_info['owners']
            ]

        return {k: v for k, v in formatted.items() if v is not None}


def create_drive_service(credentials_path: Optional[str] = None, token_path: Optional[str] = None) -> GoogleDriveService:
    """Factory function to create a Google Drive service instance"""
    return GoogleDriveService(credentials_path, token_path)

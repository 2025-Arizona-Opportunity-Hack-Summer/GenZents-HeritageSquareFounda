import os
import io
import json
from typing import Dict, List, Optional, Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google.oauth2 import service_account
from config import settings
from utils.logger import logger
import fitz


class GoogleDriveService:
    SCOPES = [
        'https://www.googleapis.com/auth/drive'
    ]

    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path or settings.GOOGLE_CREDENTIALS_PATH
        self.service = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Google Drive API using service account"""
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}")

        try:
            # Load and validate the service account credentials
            with open(self.credentials_path, 'r') as f:
                creds_info = json.load(f)

            # Verify it's a service account credentials file
            if creds_info.get('type') != 'service_account':
                raise ValueError(f"Invalid credentials type. Expected 'service_account', got '{creds_info.get('type')}'. ""Please use a service account credentials file for backend applications.")

            logger.info("Using service account authentication")
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=self.SCOPES)

            self.service = build('drive', 'v3', credentials=credentials)
            logger.info("Google Drive service initialized successfully")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in credentials file: {e}")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise RuntimeError(
                f"Failed to authenticate with Google Drive API: {e}")

    def get_file_info(self, file_id: str,
                      fields: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a file"""
        try:
            default_fields = 'id,name,mimeType,size,createdTime,modifiedTime,parents,webViewLink,webContentLink,owners,permissions'
            fields = fields or default_fields

            file_info = self.service.files().get(fileId=file_id, fields=fields).execute()
            return self._format_file_info(file_info)
        except HttpError as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            raise

    def get_file_info(self, file_id: str,
                      fields: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a file"""
        try:
            default_fields = 'id,name,mimeType,size,createdTime,modifiedTime,parents,webViewLink,webContentLink,owners,permissions'
            fields = fields or default_fields

            file_info = self.service.files().get(fileId=file_id, fields=fields).execute()
            return self._format_file_info(file_info)
        except HttpError as e:
            logger.error(f"Failed to get file info for {file_id}: {e}")
            raise

    def list_all_file_metadata(self, folder_id:Optional[str]=None):
        results = None
        if not folder_id:
            # if folder_id not provided, list all files and folders in the google drive
            results = self.service.files().list(
                pageSize=10,
                fields="nextPageToken, files(id, name, mimeType, createdTime, modifiedTime, owners)"
            ).execute()
            
        else:
            query = f"'{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType)"
            ).execute()
            files = results.get('files', [])
            print("\nFiles in folder:")
            for file in files:
                print(f"- {file['name']} ({file['mimeType']})")
        
        items = results.get('files', [])
        if not items:
            print("No files found.")
        else:
            print("Files:")
            for item in items:
                print(f"{item['name']} ({item['id']})")
                print(f"  Type: {item['mimeType']}")
                print(f"  Created: {item['createdTime']}")
                print(f"  Modified: {item['modifiedTime']}")
                print(f"  Owner: {item['owners'][0]['emailAddress']}")
                print()
        return items

    def create_folder(
            self, name: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
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

    def download_and_get_file_content(self, file_id: str, file_mimeType: str) -> bytes:
        """Download and get content of the file with file_id"""
        try:
            # process Google workspace files
            export_mime_map = {
            "application/vnd.google-apps.document": "text/plain",
            "application/vnd.google-apps.spreadsheet": "text/csv",
            "application/vnd.google-apps.presentation": "text/plain"
        }

            if file_mimeType.startswith("application/vnd.google-apps"):
                export_mime = export_mime_map[file_mimeType]
                if file_mimeType not in export_mime_map:
                    logger.warning(f"Unsupported Google type: {export_mime}")
                    return None
                
                request = self.service.files().export(fileId=file_id, mimeType=file_mimeType)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    status, done = downloader.next_chunk()

                fh.seek(0)
                content = fh.read().decode('utf-8')
                logger.info(
                    f"Downloaded file (ID: {file_id}), size: {len(content)} bytes")
                return content
            
            # process other file types
            else:
                request = self.service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                fh.seek(0)
                
                if "pdf" in file_mimeType:
                    doc = fitz.open(stream=fh, filetype="pdf")
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    print(text)
                    return text
                if "csv" in file_mimeType:
                    csv_text = fh.read().decode('utf-8', errors='ignore')
                    return csv_text
                if "text" in file_mimeType:
                    text = fh.read().decode('utf-8')
                    return text                    
                
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

    def move_file(self, file_id: str, new_parent_id: str,
                  old_parent_id: Optional[str] = None) -> Dict[str, Any]:
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

    def search_files(self, query: str,
                     max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for files by name or content"""
        try:
            search_query = f"name contains '{query}' or fullText contains '{query}'"
            return self.list_files(query=search_query, max_results=max_results)
        except HttpError as e:
            logger.error(f"Failed to search files with query '{query}': {e}")
            raise

    def get_folder_structure(
            self, folder_id: Optional[str] = None, max_depth: int = 3) -> Dict[str, Any]:
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

    def organize_by_type(
            self, source_folder_id: str) -> Dict[str, List[Dict[str, Any]]]:
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


def create_drive_service(
        credentials_path: Optional[str] = None) -> GoogleDriveService:
    """Factory function to create a Google Drive service instance"""
    return GoogleDriveService(credentials_path)

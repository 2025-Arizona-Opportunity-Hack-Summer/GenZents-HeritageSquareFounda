from typing import Optional
from googleapiclient.discovery import Resource

def get_folder_id_by_name(service: Resource, folder_name: str) -> Optional[str]:
    """Helper: Return folder ID given a name."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])
    return folders[0]['id'] if folders else None

def move_drive_file(service: Resource, file_id: str, destination_folder_name: str) -> str:
    """
    Tool: Move a file to another folder in Google Drive.

    Args:
        service: Authenticated Google Drive API service.
        file_id (str): ID of the file to move.
        destination_folder_name (str): Name of the target folder.

    Returns:
        str: Status message.
    """
    try:
        # Get the destination folder ID
        destination_folder_id = get_folder_id_by_name(service, destination_folder_name)
        if not destination_folder_id:
            return f"❌ Folder '{destination_folder_name}' not found."

        # Get the file's current parents
        file = service.files().get(fileId=file_id, fields='parents').execute()
        current_parents = ",".join(file.get('parents', []))

        # Move the file
        service.files().update(
            fileId=file_id,
            addParents=destination_folder_id,
            removeParents=current_parents,
            fields='id, parents'
        ).execute()

        return f"✅ File {file_id} moved to '{destination_folder_name}'."

    except Exception as e:
        return f"❌ Error moving file: {e}"
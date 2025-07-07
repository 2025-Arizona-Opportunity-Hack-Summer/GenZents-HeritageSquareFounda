import re

def handle_summarize_request(query: str, service) -> str:
    """
    Natural language handler to summarize a PDF file from Drive.

    Args:
        query (str): User's natural language request
        service: Google Drive API service

    Returns:
        str: Summary or error message
    """
    # Try to extract file name ending in .pdf from query
    match = re.search(r'([\w\- ]+\.pdf)', query, re.IGNORECASE)
    if not match:
        return "❌ I couldn't find a PDF file name in your request."

    file_name = match.group(1).strip()

    try:
        # Look up file in Drive
        results = service.files().list(
            q=f"name='{file_name}' and trashed=false",
            fields="files(id, name)"
        ).execute()

        files = results.get("files", [])
        if not files:
            return f"❌ File '{file_name}' not found in Drive."

        file_id = files[0]['id']
        return handle_summarize_request(service, file_id)

    except Exception as e:
        return f"❌ Something went wrong: {e}"
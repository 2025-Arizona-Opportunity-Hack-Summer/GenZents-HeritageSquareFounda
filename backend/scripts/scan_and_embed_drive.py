import os
import io
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from googleapiclient.http import MediaIoBaseDownload

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


chroma_client = PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="drive-docs")



def download_pdf_text(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    doc = fitz.open(stream=fh, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_folder_id(service, folder_name):
    response = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    folders = response.get('files', [])
    if not folders:
        raise Exception(f"Folder '{folder_name}' not found.")
    return folders[0]['id']

def list_folder_contents(service, folder_id):
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def embed_and_store_pdf(service, file):
    file_id = file['id']
    file_name = file['name']

    existing = collection.get(include=["metadatas"])
    existing_file_ids = {meta["file_id"] for meta in existing["metadatas"]}

    if file_id in existing_file_ids:
        print(f"🔁 Skipping already embedded file: {file_name}")
        return

    try:
        text = download_pdf_text(service, file_id)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        embeddings = embedding_model.encode(chunks).tolist()
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{file_id}_{idx}"],
                metadatas=[{"file_id": file_id, "file_name": file_name}]
            )
        print(f"✅ Embedded '{file_name}' ({len(chunks)} chunks)")
    except Exception as e:
        print(f"❌ Failed to embed '{file_name}': {e}")

def process_folder_recursively(service, folder_id):
    items = list_folder_contents(service, folder_id)
    for item in items:
        if item['mimeType'] == 'application/pdf':
            embed_and_store_pdf(service, item)
        elif item['mimeType'] == 'application/vnd.google-apps.folder':
            print(f"📂 Entering subfolder: {item['name']}")
            process_folder_recursively(service, item['id'])
        else:
            print(f"⏭️ Skipping non-PDF: {item['name']}")

def scan_folder_and_embed(service, folder_name):
    folder_id = get_folder_id(service, folder_name)
    print(f"📁 Scanning folder '{folder_name}' recursively...")
    process_folder_recursively(service, folder_id)
    print("💾 Chroma DB persisted to disk.")
    print("✅ Total chunks stored:", collection.count())
import os
import io
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from googleapiclient.http import MediaIoBaseDownload
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from scripts.google_drive import GoogleDriveService
from langchain import hub
from scripts.drive_agent import list_file_metadata

class RAG():
    _self = None
    def __new__(cls,*args, **kwargs):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self
    
    def __init__(self, DriveService: GoogleDriveService):
        # pass in Google Drive object (should be a singleton)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = PersistentClient(path="./chroma_store")
        self.drive = DriveService
        self.vector_store = self.initialize_vector_store()
    
    def initialize_vector_store(self):
        # read file content of the google native files and normal files
        # if the file listed is a folder --> recursively read the folder with list_files(folderId)
        vector_store = self.chroma_client.get_or_create_collection(name="drive-docs")
        files = self.drive.list_all_file_metadata()
        all_text = []
        for file in files:
            if file['mimeType'] != "application/vnd.google-apps.folder":
                text = self.drive.download_and_get_file_content(file['id'], file['mimeType'])
                all_text.append(text)
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                embeddings = self.embedding_model.encode(chunks).tolist()
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_store.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        ids=[f"{file['id']}_{idx}"],
                        metadatas=[{"file_id": file['id'], "file_name": file['name']}]
                    )
                print(f" Embedded '{file['name']}' ({len(chunks)} chunks)")
        return vector_store

    def retrieve(self,query):
        retrieved_docs = self.vector_store.query(query_texts=query)
        return retrieved_docs
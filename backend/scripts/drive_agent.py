import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from scan_and_embed_drive import scan_folder_and_embed
from dotenv import load_dotenv
from services.search_drive import search_drive_documents
from services.generate_answer_with_llm import generate_answer_with_llm
from services.move_drive_file import move_drive_file
from services.summarize_drive_file import handle_summarize_request

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/drive']

CREDENTIALS_FILE = 'client_secret_634158695245-ncmbo57t3e61ubjfoqaa7c422tpdnoti.apps.googleusercontent.com.json'


def authenticate():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def get_folder_id(service, folder_name):
    results = service.files().list(
        q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false",
        spaces='drive',
        fields="files(id, name)",
    ).execute()
    folders = results.get('files', [])
    if not folders:
        raise Exception(f"No folder found with name '{folder_name}'")
    return folders[0]['id']

def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)"
    ).execute()
    files = results.get('files', [])
    print("\nFiles in folder:")
    for file in files:
        print(f"- {file['name']} ({file['mimeType']})")

if __name__ == '__main__':
    service = authenticate()
    folder_name = 'OHacksMirror' 
    folder_id = get_folder_id(service, folder_name)
    list_files_in_folder(service, folder_id)


    print("\n🔍 Ask your Drive anything (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting agent. Bye!")
            break

        if "summarize" in query.lower() and ".pdf" in query.lower():
            summary = handle_summarize_request(query, service)
            print("\n📄 Summary:\n", summary)
            continue 

        chunks = search_drive_documents(query)
        answer = generate_answer_with_llm(query, chunks)
        print("\n🤖 Agent:\n", answer)

        move = input("Do you want to move a file from your Drive? (y/n): ")
        if move.lower() == "y":
            
            file_name = input("Enter the EXACT name of the file you want to move: ").strip()

            
            file_results = service.files().list(
                q=f"name='{file_name}' and mimeType!='application/vnd.google-apps.folder' and trashed=false",
                spaces='drive',
                fields="files(id, name)"
            ).execute()
            matched_files = file_results.get("files", [])

            if not matched_files:
                print(f"❌ No file found with name '{file_name}'")
                continue

            file_id = matched_files[0]['id']
            print(f"✅ Found file: {matched_files[0]['name']} (ID: {file_id})")

            destination = input("Enter the name of the folder to move it to: ").strip()
            result = move_drive_file(service, file_id, destination)
            print(result)

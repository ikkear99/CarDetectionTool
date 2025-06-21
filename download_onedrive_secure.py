import os
import requests
import msal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.environ.get("MS_CLIENT_ID")
TENANT_ID = os.environ.get("MS_TENANT_ID")
USERNAME = os.environ.get("MS_USERNAME")
PASSWORD = os.environ.get("MS_PASSWORD")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "video/demovideo/")

# List of OneDrive file IDs or URLs (you need to fill this)
ONEDRIVE_FILE_IDS = [
    # Example: "01ABCDEF23456789..."
    # You can also put your SharePoint/OneDrive file ID here
]

# Microsoft Graph API endpoints
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"


def get_access_token():
    app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
    result = app.acquire_token_by_username_password(
        USERNAME, PASSWORD, scopes=SCOPE
    )
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Failed to get token: {result}")


def download_onedrive_file(file_id, access_token, save_dir):
    headers = {"Authorization": f"Bearer {access_token}"}
    # Get file metadata to get the name
    meta_url = f"{GRAPH_API_BASE}/me/drive/items/{file_id}"
    meta_resp = requests.get(meta_url, headers=headers)
    if meta_resp.status_code != 200:
        print(f"Failed to get metadata for {file_id}")
        return
    filename = meta_resp.json().get("name", f"{file_id}.mp4")
    # Download file content
    download_url = f"{GRAPH_API_BASE}/me/drive/items/{file_id}/content"
    resp = requests.get(download_url, headers=headers, stream=True)
    if resp.status_code == 200:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {filename}")


def main():
    access_token = get_access_token()
    for file_id in ONEDRIVE_FILE_IDS:
        download_onedrive_file(file_id, access_token, VIDEO_DIR)

if __name__ == "__main__":
    main()

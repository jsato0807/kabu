import os
import glob
import pickle
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import time
from googleapiclient.errors import HttpError


# アップロードするCSVファイルのパス
csv_folder_path = './csv_dir/*.csv'  # ここを変更

# Google Drive APIのスコープ
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_drive():
    """Google Drive APIに認証します。"""
    creds = None
    # トークンファイルが存在する場合は、認証情報を読み込みます。
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # 認証情報が無い場合、または無効な場合は、ログインを行います。
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret_131886500269-u0ool5c31q2djp7ihr5ah0irrnk9r1cr.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # 認証情報を保存
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return creds

def upload_csv_files(creds):
    """CSVファイルをGoogle Driveにアップロードします。"""
    service = build('drive', 'v3', credentials=creds)
    
    # CSVファイルを取得
    csv_files = glob.glob(csv_folder_path)
    
    for csv_file in csv_files:
        file_metadata = {'name': os.path.basename(csv_file)}
        media = MediaFileUpload(csv_file, mimetype='text/csv')
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f'Uploaded {csv_file} to Google Drive.')

if __name__ == '__main__':
    creds = authenticate_drive()
    upload_csv_files(creds)

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
    service = build('drive', 'v3', credentials=creds)
    csv_files = glob.glob(csv_folder_path)
    csv_files = csv_files[4:]

    for csv_file in csv_files:
        if csv_file.endswith('.csv'):
            file_metadata = {'name': os.path.basename(csv_file)}
            media = MediaFileUpload(csv_file, mimetype='text/csv', resumable=True)
            
            request = service.files().create(body=file_metadata, media_body=media, fields='id')
            
            # リトライ機能付きでアップロード
            for attempt in range(5):  # 最大5回リトライ
                try:
                    print(f'Starting upload of {csv_file} (Attempt {attempt + 1})')
                    response = None
                    while response is None:
                        status, response = request.next_chunk()
                        if status:
                            print(f"Uploaded {int(status.progress() * 100)}% of {csv_file}")
                    print(f'Upload of {csv_file} completed with file ID: {response.get("id")}')
                    break  # 成功した場合ループを抜ける
                except HttpError as error:
                    print(f'HttpError uploading {csv_file}: {error}')
                    if attempt < 4:
                        print("Retrying after HttpError...")
                        time.sleep(2 ** attempt)  # 再試行前に指数バックオフを適用
                    else:
                        print("Failed to upload after multiple attempts due to HttpError.")
                except TimeoutError as timeout_error:
                    print(f'TimeoutError uploading {csv_file}: {timeout_error}')
                    if attempt < 4:
                        print("Retrying due to TimeoutError...")
                        time.sleep(2 ** attempt)
                    else:
                        print("Failed to upload after multiple timeout attempts.")

if __name__ == '__main__':
    creds = authenticate_drive()
    upload_csv_files(creds)

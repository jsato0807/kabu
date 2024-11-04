import os
import glob
import pickle
import zipfile
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import time
from googleapiclient.errors import HttpError
import ssl


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

def zip_csv_file(csv_file):
    """CSVファイルをZIPファイルに圧縮します。"""
    zip_filename = f"{os.path.splitext(csv_file)[0]}.zip"  # 拡張子を除いてZIPファイル名を生成
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        zip_file.write(csv_file, os.path.basename(csv_file))  # CSVファイルをZIPに追加
    return zip_filename

def upload_csv_files(creds):
    service = build('drive', 'v3', credentials=creds)
    csv_files = glob.glob(csv_folder_path)

    if not csv_files:
        print("No CSV files found to upload.")
        return

    for csv_file in csv_files:
        zip_filename = zip_csv_file(csv_file)  # 各CSVファイルをZIPに圧縮
        
        file_metadata = {'name': os.path.basename(zip_filename)}  # ZIPファイル名のみを使用
        media = MediaFileUpload(zip_filename, mimetype='application/zip', resumable=True)
        
        # リトライ機能付きアップロード
        for attempt in range(5):  # 最大5回リトライ
            try:
                # 再試行時に新しいリクエストを生成
                request = service.files().create(body=file_metadata, media_body=media, fields='id')
                
                print(f'Starting upload of {zip_filename} (Attempt {attempt + 1})')
                response = None
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        print(f"Uploaded {int(status.progress() * 100)}% of {zip_filename}")
                print(f'Upload of {zip_filename} completed with file ID: {response.get("id")}')
                break  # 成功した場合ループを抜ける
            except HttpError as error:
                print(f'HttpError uploading {zip_filename}: {error}')
                if attempt < 4:
                    print("Retrying after HttpError...")
                    time.sleep(2 ** attempt)
                else:
                    print("Failed to upload after multiple attempts due to HttpError.")
            except ssl.SSLEOFError as ssl_error:
                print(f'SSLEOFError uploading {zip_filename}: {ssl_error}')
                if attempt < 4:
                    print("Retrying due to SSLEOFError...")
                    time.sleep(2 ** attempt)
                else:
                    print("Failed to upload after multiple attempts due to SSLEOFError.")

if __name__ == '__main__':
    creds = authenticate_drive()
    upload_csv_files(creds)

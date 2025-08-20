import sys
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
REST_API_KEY = os.getenv('REST_API_KEY')
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

TO_URL = 'https://kapi.kakao.com/v2/api/talk/memo/default/send'
TOKEN_URL = 'https://kauth.kakao.com/oauth/token'

def refresh_access_token():
    data = {
        'grant_type': 'refresh_token',
        'client_id': REST_API_KEY,
        'refresh_token': REFRESH_TOKEN
    }

    response = requests.post(TOKEN_URL, data=data)
    if response.status_code == 200:
        token_json = response.json()
        new_token = token_json['access_token']
        print("[INFO] 새 access token 발급 성공:", new_token)
        return new_token
    else:
        print("[ERROR] access token 갱신 실패:", response.status_code, response.text)
        return None

def send_kakao_message(url, token):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    template_object = {
        "object_type": "text",
        "text": f"[NGROK 연결주소]\n{url}",
        "link": {
            "web_url": url,
            "mobile_web_url": url
        }
    }

    data = {
        'template_object': json.dumps(template_object)
    }

    response = requests.post(TO_URL, headers=headers, data=data)
    return response

if __name__ == "__main__":
    url_to_send = sys.argv[1]

    print("[INFO] 1차 시도: access token으로 전송")
    response = send_kakao_message(url_to_send, ACCESS_TOKEN)

    if response.status_code == 401: 
        print("[WARN] access token 만료. refresh token으로 새 토큰 발급 시도")
        new_token = refresh_access_token()
        if new_token:
            response = send_kakao_message(url_to_send, new_token)

    print("카카오톡 전송 응답:", response.status_code, response.text)

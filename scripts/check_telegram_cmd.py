#!/usr/bin/env python3
"""
Telegram명령어 감지 스크립트
Watchdog이 3분마다 이 스크립트를 실행해, 사용자가 텔레그램으로 보낸 "/reconnect" 또는 "/rc" 명령어를 감지합니다.

명령어가 발견되면 exit code 0을 반환 (Watchdog이 즉시 재시작 실행)
명령어가 없거나 에러가 나면 exit code 1을 반환
"""

import os
import sys
import json
import logging
from pathlib import Path

# VENV 로드나 의존성 이슈 방지를 위해 내장 모듈 사용
# 만약 requests가 없다면 urllib.request로 폴백
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    USE_REQUESTS = False

PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_DIR / ".env"
UPDATE_ID_FILE = PROJECT_DIR / "data" / ".tg_update_id"

def get_env_var(key: str) -> str:
    if not ENV_FILE.exists():
        return ""
    with open(ENV_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip('"\'')
    return ""

def main():
    bot_token = get_env_var("TELEGRAM_BOT_TOKEN")
    user_id = get_env_var("TELEGRAM_USER_ID")

    if not bot_token or not user_id:
        sys.exit(1)

    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    
    # 마지막으로 처리한 update_id 가져오기
    last_update_id = 0
    if UPDATE_ID_FILE.exists():
        try:
            with open(UPDATE_ID_FILE, "r") as f:
                last_update_id = int(f.read().strip())
        except Exception:
            pass

    # 파라미터 구성 (이미 확인한 메시지는 무시하도록 offset 설정)
    query = ""
    if last_update_id > 0:
        query = f"?offset={last_update_id + 1}"

    try:
        reqUrl = url + query
        req = urllib.request.Request(reqUrl)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        if not data or not data.get("ok"):
            sys.exit(1)

        updates = data.get("result", [])
        should_reconnect = False
        highest_update_id = last_update_id

        for update in updates:
            up_id = update.get("update_id", 0)
            if up_id > highest_update_id:
                highest_update_id = up_id

            msg = update.get("message", {})
            sender_id = str(msg.get("from", {}).get("id", ""))
            
            # 본인이 보낸 메시지만 허용
            if sender_id == str(user_id):
                text = msg.get("text", "").strip().lower()
                if text in ["/reconnect", "/rc"]:
                    should_reconnect = True
                    
                # 이미지/사진 파일이 전송된 경우 바탕화면으로 다운로드 (Claude가 볼 수 있게)
                photo_array = msg.get("photo", [])
                if photo_array:
                    # 보통 가장 큰 사이즈의 사진이 배열 마지막에 있음
                    best_photo = photo_array[-1]
                    file_id = best_photo.get("file_id")
                    if file_id:
                        getFile_url = f"https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}"
                        try:
                            req_file = urllib.request.Request(getFile_url)
                            with urllib.request.urlopen(req_file, timeout=10) as r_file:
                                file_data = json.loads(r_file.read().decode())
                                file_path = file_data.get("result", {}).get("file_path")
                                if file_path:
                                    dl_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
                                    req_dl = urllib.request.Request(dl_url)
                                    with urllib.request.urlopen(req_dl, timeout=15) as r_dl:
                                        img_bytes = r_dl.read()
                                        timestamp = msg.get("date", 0)
                                        save_path = PROJECT_DIR / "logs" / f"telegram_photo_{timestamp}.jpg"
                                        with open(save_path, "wb") as img_file:
                                            img_file.write(img_bytes)
                                        
                                        # tmux 세션에 클로드가 인지하도록 메시지 전송
                                        os.system(f"tmux send-keys -t blockchain:claude '사용자가 텔레그램을 통해 아이폰 스크린샷 1장을 전송했습니다. 확인해주세요: {save_path}' Enter")
                        except Exception as e:
                            pass # 이미지 다운로드 실패 무시

        # update_id 저장 (다음 번엔 이 메시지들을 무시함)
        if highest_update_id > last_update_id:
            with open(UPDATE_ID_FILE, "w") as f:
                f.write(str(highest_update_id))

        if should_reconnect:
            sys.exit(0)  # 리커넥트 명령 감지됨!
        else:
            sys.exit(1)  # 명령 없음

    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()

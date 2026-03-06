#!/usr/bin/env python3
"""
텔레그램 알림 전송 스크립트

메시지 타입: trade, analysis, error, status
포맷: MarkdownV2

사용법:
  python3 scripts/notify_telegram.py trade "BTC 매수 실행" "10만원 매수, RSI 28"
  python3 scripts/notify_telegram.py error "데이터 수집 실패" "Upbit API 타임아웃"

출력: JSON (stdout)
"""

import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TELEGRAM_API = "https://api.telegram.org/bot{token}"

EMOJI = {
    "trade": "\U0001f4b0",     # 💰
    "analysis": "\U0001f4ca",  # 📊
    "error": "\U0001f6a8",     # 🚨
    "status": "\U0001f4cb",    # 📋
}

KST = timezone(timedelta(hours=9))


def escape_md(text: str) -> str:
    """MarkdownV2 특수문자 이스케이프"""
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!\\])", r"\\\1", text)


def send_message(msg_type: str, title: str, body: str):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    user_id = os.environ.get("TELEGRAM_USER_ID")
    if not bot_token or not user_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN 또는 TELEGRAM_USER_ID 미설정")

    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    emoji = EMOJI.get(msg_type, "\U0001f4ac")
    text = f"{emoji} *{escape_md(title)}*\n\n{escape_md(body)}\n\n_{escape_md(ts)}_"

    r = requests.post(
        f"{TELEGRAM_API.format(token=bot_token)}/sendMessage",
        json={
            "chat_id": user_id,
            "text": text,
            "parse_mode": "MarkdownV2",
        },
        timeout=10,
    )

    if not r.ok:
        raise RuntimeError(f"텔레그램 전송 실패: {r.text}")

    return {"success": True, "type": msg_type, "title": title}


def send_photo(image_path: str, caption: str):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    user_id = os.environ.get("TELEGRAM_USER_ID")
    if not bot_token or not user_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN 또는 TELEGRAM_USER_ID 미설정")

    with open(image_path, "rb") as f:
        r = requests.post(
            f"{TELEGRAM_API.format(token=bot_token)}/sendPhoto",
            data={"chat_id": user_id, "caption": caption},
            files={"photo": ("chart.png", f, "image/png")},
            timeout=30,
        )

    if not r.ok:
        raise RuntimeError(f"텔레그램 이미지 전송 실패: {r.text}")

    return {"success": True, "type": "photo", "path": image_path}


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            '사용법: python3 notify_telegram.py [trade|analysis|error|status] "제목" "본문"',
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = send_message(sys.argv[1], sys.argv[2], sys.argv[3])
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)

"""Kimchirang Notifier -- 텔레그램 알림 (구조화 메시지)"""

import asyncio
import logging
import os
import sys

logger = logging.getLogger("kimchirang.notifier")

# 프로젝트 루트 → scripts import 경로
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


class KimchirangNotifier:
    """차익거래 전용 텔레그램 알림"""

    def __init__(self):
        self._enabled = bool(
            os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_USER_ID")
        )
        if not self._enabled:
            logger.warning("텔레그램 미설정 -- 알림 비활성화")

    async def _send(self, msg_type: str, title: str, body: str):
        """비동기 텔레그램 전송 (기존 notify_telegram.py 재사용)"""
        if not self._enabled:
            return
        try:
            from scripts.notify_telegram import send_message
            await asyncio.to_thread(send_message, msg_type, title, body)
        except Exception as e:
            logger.error(f"텔레그램 전송 실패: {e}")

    async def notify_enter(self, result, snapshot):
        """진입 알림"""
        dry = "[DRY] " if result.dry_run else ""
        body = (
            f"KP: {snapshot.entry_kp:+.2f}% (Entry)\n"
            f"Upbit: {result.upbit_leg.filled_price:,.0f} KRW\n"
            f"Binance: {result.binance_leg.filled_price:,.2f} USDT\n"
            f"수량: {result.upbit_leg.filled_qty:.8f} BTC\n"
            f"FX: {snapshot.fx_rate:,.2f}\n"
            f"Latency: {result.total_latency_ms:.0f}ms"
        )
        await self._send("trade", f"{dry}Kimchirang 진입", body)

    async def notify_exit(self, result, snapshot, pnl: float):
        """청산 알림"""
        dry = "[DRY] " if result.dry_run else ""
        emoji = "+" if pnl >= 0 else ""
        body = (
            f"PnL: {emoji}{pnl:.2f}%\n"
            f"KP: {snapshot.exit_kp:+.2f}% (Exit)\n"
            f"Upbit: {result.upbit_leg.filled_price:,.0f} KRW\n"
            f"Binance: {result.binance_leg.filled_price:,.2f} USDT\n"
            f"Latency: {result.total_latency_ms:.0f}ms"
        )
        await self._send("trade", f"{dry}Kimchirang 청산 ({emoji}{pnl:.2f}%)", body)

    async def notify_stop_loss(self, result, snapshot, pnl: float):
        """손절 알림"""
        dry = "[DRY] " if result.dry_run else ""
        body = (
            f"손절 PnL: {pnl:+.2f}%\n"
            f"KP: {snapshot.mid_kp:+.2f}% (손절 기준 {8.0}%)\n"
            f"Latency: {result.total_latency_ms:.0f}ms"
        )
        await self._send("error", f"{dry}Kimchirang 손절!", body)

    async def notify_error(self, phase: str, error: str):
        """에러 알림"""
        body = f"Phase: {phase}\nError: {error[:300]}"
        await self._send("error", "Kimchirang 오류", body)

    async def notify_status(self, stats: dict, position: dict):
        """주기적 상태 보고"""
        side = position.get("side", "none")
        kp = stats.get("mid_kp", 0)
        z = stats.get("kp_z_score", 0)
        vel = stats.get("kp_velocity", 0)
        body = (
            f"KP: {kp:+.2f}% | Z: {z:+.2f} | V: {vel:+.3f}%/m\n"
            f"포지션: {side}\n"
            f"금일 거래: {position.get('trades_today', 0)}회"
        )
        if position.get("is_open"):
            body += (
                f"\n진입 KP: {position.get('entry_kp', 0):+.2f}%"
                f"\n보유: {position.get('hold_duration_min', 0):.0f}분"
            )
        await self._send("status", "Kimchirang 상태", body)

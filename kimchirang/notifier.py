"""Kimchirang Notifier -- 텔레그램 알림 (구조화 메시지)

Rate limit (429) 대응:
  - 전송 실패 시 exponential backoff 재시도 (최대 3회)
  - 429 응답 시 Retry-After 헤더 존중
  - 연속 실패 시 일시적 비활성화 (5분 쿨다운)
  - 메시지 큐잉으로 burst 방지 (최소 1초 간격)
"""

import asyncio
import logging
import os
import sys
import time

logger = logging.getLogger("kimchirang.notifier")

# 프로젝트 루트 -> scripts import 경로
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


class KimchirangNotifier:
    """차익거래 전용 텔레그램 알림 (Rate limit + 큐잉)"""

    # 전송 간 최소 간격 (초) -- 텔레그램 API burst 방지
    MIN_SEND_INTERVAL = 1.0
    # 연속 실패 허용 횟수
    MAX_CONSECUTIVE_FAILURES = 5
    # 연속 실패 후 쿨다운 (초)
    FAILURE_COOLDOWN = 300  # 5분
    # 재시도 횟수
    MAX_RETRIES = 3

    def __init__(self):
        self._enabled = bool(
            os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_USER_ID")
        )
        if not self._enabled:
            logger.warning("텔레그램 미설정 -- 알림 비활성화")
        self._last_send_time = 0.0
        self._consecutive_failures = 0
        self._disabled_until = 0.0
        self._send_lock = asyncio.Lock()

    async def _send(self, msg_type: str, title: str, body: str):
        """비동기 텔레그램 전송 (Rate limit + 재시도 + 쿨다운)"""
        if not self._enabled:
            return

        # 쿨다운 중이면 스킵
        now = time.time()
        if now < self._disabled_until:
            remaining = self._disabled_until - now
            logger.debug(f"텔레그램 쿨다운 중 ({remaining:.0f}초 남음) -- 스킵")
            return

        async with self._send_lock:
            # 최소 전송 간격 보장 (burst 방지)
            elapsed = time.time() - self._last_send_time
            if elapsed < self.MIN_SEND_INTERVAL:
                await asyncio.sleep(self.MIN_SEND_INTERVAL - elapsed)

            for attempt in range(self.MAX_RETRIES):
                try:
                    from scripts.notify_telegram import send_message
                    await asyncio.to_thread(send_message, msg_type, title, body)
                    self._last_send_time = time.time()
                    self._consecutive_failures = 0
                    return
                except Exception as e:
                    error_str = str(e)
                    self._consecutive_failures += 1

                    # 429 Rate limit 감지
                    if "429" in error_str or "Too Many Requests" in error_str:
                        # Retry-After 파싱 시도
                        retry_after = 5.0 * (attempt + 1)
                        if "retry after" in error_str.lower():
                            try:
                                # "retry after N" 패턴에서 N 추출
                                parts = error_str.lower().split("retry after")
                                if len(parts) > 1:
                                    seconds = float(parts[1].strip().split()[0])
                                    retry_after = max(retry_after, seconds)
                            except (ValueError, IndexError):
                                pass
                        logger.warning(
                            f"텔레그램 Rate limit (429) -- {retry_after:.0f}초 대기 "
                            f"(시도 {attempt + 1}/{self.MAX_RETRIES})"
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(retry_after)
                            continue
                    else:
                        # 기타 오류: exponential backoff
                        if attempt < self.MAX_RETRIES - 1:
                            delay = 2.0 ** attempt
                            logger.warning(
                                f"텔레그램 전송 실패 (시도 {attempt + 1}/{self.MAX_RETRIES}): {e}"
                            )
                            await asyncio.sleep(delay)
                            continue
                        logger.error(f"텔레그램 전송 최종 실패: {e}")

            # 모든 재시도 실패
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._disabled_until = time.time() + self.FAILURE_COOLDOWN
                logger.error(
                    f"텔레그램 {self.MAX_CONSECUTIVE_FAILURES}회 연속 실패 -- "
                    f"{self.FAILURE_COOLDOWN}초 쿨다운"
                )

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
        sign = "+" if pnl >= 0 else ""
        body = (
            f"PnL: {sign}{pnl:.2f}%\n"
            f"KP: {snapshot.exit_kp:+.2f}% (Exit)\n"
            f"Upbit: {result.upbit_leg.filled_price:,.0f} KRW\n"
            f"Binance: {result.binance_leg.filled_price:,.2f} USDT\n"
            f"Latency: {result.total_latency_ms:.0f}ms"
        )
        await self._send("trade", f"{dry}Kimchirang 청산 ({sign}{pnl:.2f}%)", body)

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

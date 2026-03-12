"""Kimchirang DB -- Supabase REST API를 통한 차익거래 기록"""

import asyncio
import logging
from typing import Optional

import requests

from kimchirang.config import DBConfig
from kimchirang.execution import ExecutionResult
from kimchirang.kp_engine import KPSnapshot

logger = logging.getLogger("kimchirang.db")


class KimchirangDB:
    """Supabase PostgREST로 차익거래 기록"""

    def __init__(self, config: DBConfig):
        self._url = config.supabase_url
        self._key = config.supabase_key
        self._enabled = bool(self._url and self._key)
        if not self._enabled:
            logger.warning("Supabase 미설정 -- DB 기록 비활성화")

    def _post(self, table: str, row: dict) -> bool:
        """동기 POST (asyncio.to_thread에서 호출)"""
        if not self._enabled:
            return False
        try:
            resp = requests.post(
                f"{self._url}/rest/v1/{table}",
                json=row,
                headers={
                    "apikey": self._key,
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                timeout=10,
            )
            if resp.status_code in (200, 201):
                return True
            logger.error(f"DB 기록 실패 ({table}): {resp.status_code} {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"DB 기록 오류 ({table}): {e}")
            return False

    async def record_trade(
        self,
        result: ExecutionResult,
        snapshot: KPSnapshot,
        pnl: Optional[float] = None,
        stats: Optional[dict] = None,
        hold_duration_min: Optional[float] = None,
    ):
        """차익거래 기록 (비동기)"""
        action = result.action
        if pnl is not None and action == "exit" and snapshot.mid_kp >= 8.0:
            action = "stop_loss"

        row = {
            "action": action,
            "kp_at_execution": result.kp_at_execution,
            "entry_kp": snapshot.entry_kp,
            "exit_kp": snapshot.exit_kp,
            "pnl_pct": pnl,
            "upbit_order_id": result.upbit_leg.order_id,
            "upbit_side": result.upbit_leg.side,
            "upbit_price": int(result.upbit_leg.filled_price) if result.upbit_leg.filled_price else None,
            "upbit_qty": result.upbit_leg.filled_qty,
            "binance_order_id": result.binance_leg.order_id,
            "binance_side": result.binance_leg.side,
            "binance_price": result.binance_leg.filled_price,
            "binance_qty": result.binance_leg.filled_qty,
            "fx_rate": snapshot.fx_rate,
            "funding_rate": snapshot.funding_rate,
            "spread_cost": (snapshot.upbit_spread_pct + snapshot.binance_spread_pct),
            "latency_ms": int(result.total_latency_ms),
            "dry_run": result.dry_run,
            "both_success": result.both_success,
            "hold_duration_min": hold_duration_min,
            "kp_stats": stats,
        }

        await asyncio.to_thread(self._post, "kimchirang_trades", row)

    async def record_error(self, phase: str, error: str):
        """에러 기록"""
        row = {
            "action": "enter",
            "kp_at_execution": 0,
            "both_success": False,
            "kp_stats": {"error_phase": phase, "error_message": error[:500]},
            "dry_run": True,
        }
        await asyncio.to_thread(self._post, "kimchirang_trades", row)

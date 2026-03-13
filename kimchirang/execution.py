"""Kimchirang Execution -- Delta-Neutral 동시 주문 실행

핵심 원칙:
  1. Upbit Spot + Binance Futures를 asyncio.gather()로 동시 실행
  2. 한쪽 실패 시 다른 쪽 긴급 청산 (레그 리스크 방지)
  3. DRY_RUN 모드에서는 실제 주문 없이 시뮬레이션
  4. API 호출 실패 시 config.trading.retry_count 만큼 재시도
"""

import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import aiohttp
import jwt

from kimchirang.config import KimchirangConfig
from kimchirang.kp_engine import KPSnapshot

logger = logging.getLogger("kimchirang.execution")

# 재시도하면 안 되는 HTTP 상태 코드 (클라이언트 오류)
_NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}


class PositionSide(Enum):
    NONE = "none"
    LONG_KP = "long_kp"   # Upbit Long + Binance Short (KP 수축 베팅)


@dataclass
class LegResult:
    """단일 레그 주문 결과"""
    exchange: str           # "upbit" or "binance"
    success: bool
    order_id: str = ""
    filled_price: float = 0.0
    filled_qty: float = 0.0
    side: str = ""          # "buy" or "sell"
    error: str = ""
    latency_ms: float = 0.0


@dataclass
class ExecutionResult:
    """양쪽 동시 주문 결과"""
    action: str             # "enter" or "exit"
    upbit_leg: LegResult = field(default_factory=lambda: LegResult(exchange="upbit", success=False))
    binance_leg: LegResult = field(default_factory=lambda: LegResult(exchange="binance", success=False))
    kp_at_execution: float = 0.0
    total_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    dry_run: bool = True

    @property
    def both_success(self) -> bool:
        return self.upbit_leg.success and self.binance_leg.success

    @property
    def partial_fill(self) -> bool:
        return self.upbit_leg.success != self.binance_leg.success

    def summary(self) -> str:
        status = "OK" if self.both_success else ("PARTIAL" if self.partial_fill else "FAIL")
        dry = "[DRY] " if self.dry_run else ""
        return (
            f"{dry}[{self.action.upper()}] {status} | "
            f"KP={self.kp_at_execution:.2f}% | "
            f"Upbit: {self.upbit_leg.filled_price:,.0f} KRW | "
            f"Binance: {self.binance_leg.filled_price:,.2f} USDT | "
            f"Latency: {self.total_latency_ms:.0f}ms"
        )


@dataclass
class PositionState:
    """현재 포지션 상태"""
    side: PositionSide = PositionSide.NONE
    entry_kp: float = 0.0
    entry_time: float = 0.0
    upbit_qty: float = 0.0       # BTC 수량 (Upbit)
    binance_qty: float = 0.0     # BTC 수량 (Binance Futures)
    upbit_entry_price: float = 0.0
    binance_entry_price: float = 0.0
    trade_count_today: int = 0
    last_trade_time: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.side != PositionSide.NONE

    @property
    def hold_duration_min(self) -> float:
        if not self.is_open:
            return 0
        return (time.time() - self.entry_time) / 60

    def to_dict(self) -> dict:
        """직렬화"""
        return {
            "side": self.side.value,
            "entry_kp": self.entry_kp,
            "entry_time": self.entry_time,
            "upbit_qty": self.upbit_qty,
            "binance_qty": self.binance_qty,
            "upbit_entry_price": self.upbit_entry_price,
            "binance_entry_price": self.binance_entry_price,
            "trade_count_today": self.trade_count_today,
            "last_trade_time": self.last_trade_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PositionState":
        """역직렬화"""
        side_str = d.get("side", "none")
        try:
            side = PositionSide(side_str)
        except ValueError:
            side = PositionSide.NONE
        return cls(
            side=side,
            entry_kp=d.get("entry_kp", 0.0),
            entry_time=d.get("entry_time", 0.0),
            upbit_qty=d.get("upbit_qty", 0.0),
            binance_qty=d.get("binance_qty", 0.0),
            upbit_entry_price=d.get("upbit_entry_price", 0.0),
            binance_entry_price=d.get("binance_entry_price", 0.0),
            trade_count_today=d.get("trade_count_today", 0),
            last_trade_time=d.get("last_trade_time", 0.0),
        )


class Executor:
    """Delta-Neutral 주문 실행기"""

    def __init__(self, config: KimchirangConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        # Binance 서버 시간 오프셋 (ms) — sync_server_time()으로 갱신
        self._binance_time_offset: int = 0

        # 포지션 영속성: 파일에서 복원
        from kimchirang.state import load_position, save_position
        self._save_position_fn = save_position
        saved = load_position()
        self.position = PositionState.from_dict(saved)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.trading.order_timeout_sec)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _binance_timestamp(self) -> str:
        """Binance 서버 시간 보정된 타임스탬프 (ms)"""
        return str(int(time.time() * 1000) + self._binance_time_offset)

    async def sync_server_time(self) -> bool:
        """Binance 서버 시간과 로컬 시간 차이를 측정하여 오프셋 저장"""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.config.binance.rest_url}/fapi/v1/time"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    server_time = data["serverTime"]
                    local_time = int(time.time() * 1000)
                    self._binance_time_offset = server_time - local_time
                    logger.info(
                        f"Binance 시간 동기화 완료 (오프셋: {self._binance_time_offset}ms)"
                    )
                    return True
        except Exception as e:
            logger.warning(f"Binance 시간 동기화 실패: {e}")
        return False

    # ============================================================
    # 안전장치 체크
    # ============================================================

    def _check_safety(self, action: str) -> Optional[str]:
        """주문 전 안전장치 확인 -- 위반 시 사유 문자열 반환"""
        if self.config.trading.emergency_stop:
            return "EMERGENCY_STOP 활성화"

        now = time.time()

        if action == "enter":
            if self.position.is_open:
                return "이미 포지션 보유 중"
            if self.position.trade_count_today >= self.config.trading.max_daily_trades:
                return f"일일 거래 한도 초과 ({self.position.trade_count_today})"
            if (now - self.position.last_trade_time) < self.config.trading.min_trade_interval_sec:
                remaining = self.config.trading.min_trade_interval_sec - (now - self.position.last_trade_time)
                return f"최소 거래 간격 미충족 ({remaining:.0f}초 남음)"

        elif action == "exit":
            if not self.position.is_open:
                return "청산할 포지션 없음"

        return None

    # ============================================================
    # 진입: Upbit Buy + Binance Short
    # ============================================================

    async def enter(self, snapshot: KPSnapshot) -> ExecutionResult:
        """Delta-Neutral 진입

        Upbit: 시장가 매수 (Spot)
        Binance: 시장가 매도/숏 (Futures 1x)
        """
        result = ExecutionResult(
            action="enter",
            kp_at_execution=snapshot.entry_kp,
            dry_run=self.config.trading.dry_run,
        )

        # 안전장치
        safety = self._check_safety("enter")
        if safety:
            logger.warning(f"진입 차단: {safety}")
            result.upbit_leg.error = safety
            result.binance_leg.error = safety
            return result

        # 주문 수량 계산
        trade_krw = self.config.trading.trade_amount_krw
        btc_price_krw = snapshot.upbit_ask
        btc_qty = trade_krw / btc_price_krw if btc_price_krw > 0 else 0

        if btc_qty <= 0:
            result.upbit_leg.error = "수량 계산 실패"
            return result

        start = time.time()

        if self.config.trading.dry_run:
            # DRY RUN: 시뮬레이션
            result.upbit_leg = LegResult(
                exchange="upbit", success=True, side="buy",
                order_id=f"dry_{uuid.uuid4().hex[:8]}",
                filled_price=snapshot.upbit_ask,
                filled_qty=btc_qty,
            )
            result.binance_leg = LegResult(
                exchange="binance", success=True, side="sell",
                order_id=f"dry_{uuid.uuid4().hex[:8]}",
                filled_price=snapshot.binance_bid,
                filled_qty=btc_qty,
            )
            logger.info(f"[DRY RUN] 진입 시뮬레이션: {btc_qty:.8f} BTC, KP={snapshot.entry_kp:.2f}%")
        else:
            # 실제 주문: 동시 실행 (retry 포함)
            upbit_task = self._upbit_market_order_with_retry("buy", btc_qty, trade_krw)
            binance_task = self._binance_futures_order_with_retry("sell", btc_qty)

            upbit_result, binance_result = await asyncio.gather(
                upbit_task, binance_task, return_exceptions=True
            )

            # 결과 처리
            result.upbit_leg = self._parse_leg_result("upbit", "buy", upbit_result)
            result.binance_leg = self._parse_leg_result("binance", "sell", binance_result)

            # 한쪽 실패 시 긴급 청산
            if result.partial_fill:
                await self._emergency_unwind(result)

        result.total_latency_ms = (time.time() - start) * 1000

        # 포지션 상태 업데이트
        if result.both_success:
            self.position.side = PositionSide.LONG_KP
            self.position.entry_kp = snapshot.entry_kp
            self.position.entry_time = time.time()
            self.position.upbit_qty = result.upbit_leg.filled_qty
            self.position.binance_qty = result.binance_leg.filled_qty
            self.position.upbit_entry_price = result.upbit_leg.filled_price
            self.position.binance_entry_price = result.binance_leg.filled_price
            self.position.trade_count_today += 1
            self.position.last_trade_time = time.time()
            self._save_state()

        logger.info(result.summary())
        return result

    # ============================================================
    # 청산: Upbit Sell + Binance Cover
    # ============================================================

    async def exit(self, snapshot: KPSnapshot) -> ExecutionResult:
        """Delta-Neutral 청산

        Upbit: 시장가 매도 (Spot)
        Binance: 시장가 매수/커버 (Futures)
        """
        result = ExecutionResult(
            action="exit",
            kp_at_execution=snapshot.exit_kp,
            dry_run=self.config.trading.dry_run,
        )

        safety = self._check_safety("exit")
        if safety:
            logger.warning(f"청산 차단: {safety}")
            result.upbit_leg.error = safety
            result.binance_leg.error = safety
            return result

        start = time.time()

        if self.config.trading.dry_run:
            result.upbit_leg = LegResult(
                exchange="upbit", success=True, side="sell",
                order_id=f"dry_{uuid.uuid4().hex[:8]}",
                filled_price=snapshot.upbit_bid,
                filled_qty=self.position.upbit_qty,
            )
            result.binance_leg = LegResult(
                exchange="binance", success=True, side="buy",
                order_id=f"dry_{uuid.uuid4().hex[:8]}",
                filled_price=snapshot.binance_ask,
                filled_qty=self.position.binance_qty,
            )
            # 수익 계산
            pnl = self._calculate_pnl(result, snapshot)
            logger.info(f"[DRY RUN] 청산 시뮬레이션: PnL={pnl:+.2f}%, KP={snapshot.exit_kp:.2f}%")
        else:
            upbit_task = self._upbit_market_order_with_retry("sell", self.position.upbit_qty)
            binance_task = self._binance_futures_order_with_retry("buy", self.position.binance_qty)

            upbit_result, binance_result = await asyncio.gather(
                upbit_task, binance_task, return_exceptions=True
            )

            result.upbit_leg = self._parse_leg_result("upbit", "sell", upbit_result)
            result.binance_leg = self._parse_leg_result("binance", "buy", binance_result)

            if result.partial_fill:
                await self._emergency_unwind(result)

        result.total_latency_ms = (time.time() - start) * 1000

        if result.both_success:
            pnl = self._calculate_pnl(result, snapshot)
            logger.info(f"청산 완료: PnL={pnl:+.2f}%, 보유시간={self.position.hold_duration_min:.1f}분")
            # pnl을 result에 저장 (포지션 초기화 후에도 참조 가능)
            result._cached_pnl = pnl
            # 포지션 초기화
            self.position.side = PositionSide.NONE
            self.position.entry_kp = 0
            self.position.upbit_qty = 0
            self.position.binance_qty = 0
            self.position.trade_count_today += 1
            self.position.last_trade_time = time.time()
            self._save_state()

        logger.info(result.summary())
        return result

    # ============================================================
    # Retry 래퍼
    # ============================================================

    async def _upbit_market_order_with_retry(
        self, side: str, qty: float, total_krw: float = None
    ) -> dict:
        """Upbit 주문 + 재시도 (config.trading.retry_count)"""
        max_retries = self.config.trading.retry_count
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = await self._upbit_market_order(side, qty, total_krw)
                if result.get("success"):
                    return result
                # API가 응답했지만 실패인 경우
                error_str = result.get("error", "")
                # 재시도 불가능한 오류 (잔고 부족, 인증 실패 등)
                if any(code in error_str for code in ["insufficient", "401", "403"]):
                    logger.error(f"Upbit 주문 실패 (재시도 불가): {error_str}")
                    return result
                last_error = error_str
            except asyncio.TimeoutError:
                last_error = "타임아웃"
                logger.warning(f"Upbit 주문 타임아웃 (시도 {attempt + 1}/{max_retries + 1})")
            except aiohttp.ClientError as e:
                last_error = str(e)
                logger.warning(f"Upbit 주문 네트워크 오류 (시도 {attempt + 1}/{max_retries + 1}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Upbit 주문 예상치 못한 오류: {type(e).__name__}: {e}")
                return {"success": False, "error": last_error}

            if attempt < max_retries:
                delay = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s...
                logger.info(f"Upbit 주문 재시도 대기 {delay:.1f}초")
                await asyncio.sleep(delay)

        return {"success": False, "error": f"재시도 {max_retries}회 실패: {last_error}"}

    async def _binance_futures_order_with_retry(
        self, side: str, qty: float
    ) -> dict:
        """Binance Futures 주문 + 재시도 (config.trading.retry_count)"""
        max_retries = self.config.trading.retry_count
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = await self._binance_futures_order(side, qty)
                if result.get("success"):
                    return result
                error_str = result.get("error", "")
                # 재시도 불가능한 오류
                if any(code in error_str for code in [
                    "insufficient", "-2019", "-2015", "-1013"
                ]):
                    logger.error(f"Binance 주문 실패 (재시도 불가): {error_str}")
                    return result
                last_error = error_str
            except asyncio.TimeoutError:
                last_error = "타임아웃"
                logger.warning(f"Binance 주문 타임아웃 (시도 {attempt + 1}/{max_retries + 1})")
            except aiohttp.ClientError as e:
                last_error = str(e)
                logger.warning(f"Binance 주문 네트워크 오류 (시도 {attempt + 1}/{max_retries + 1}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Binance 주문 예상치 못한 오류: {type(e).__name__}: {e}")
                return {"success": False, "error": last_error}

            if attempt < max_retries:
                delay = 0.5 * (2 ** attempt)
                logger.info(f"Binance 주문 재시도 대기 {delay:.1f}초")
                await asyncio.sleep(delay)

        return {"success": False, "error": f"재시도 {max_retries}회 실패: {last_error}"}

    # ============================================================
    # Upbit REST API
    # ============================================================

    async def _upbit_market_order(self, side: str, qty: float,
                                   total_krw: float = None) -> dict:
        """Upbit 시장가 주문

        매수: total_krw 기준 (원화 기준 시장가)
        매도: qty 기준 (BTC 수량 기준 시장가)
        """
        access_key = self.config.upbit.access_key
        secret_key = self.config.upbit.secret_key

        params = {
            "market": self.config.trading.upbit_symbol,
            "side": "bid" if side == "buy" else "ask",
            "ord_type": "price" if side == "buy" else "market",
        }
        if side == "buy" and total_krw:
            params["price"] = str(int(total_krw))
        elif side == "sell":
            params["volume"] = f"{qty:.8f}"

        # JWT 생성
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        m = hashlib.sha512()
        m.update(query_string.encode())
        query_hash = m.hexdigest()

        payload = {
            "access_key": access_key,
            "nonce": str(uuid.uuid4()),
            "query_hash": query_hash,
            "query_hash_alg": "SHA512",
        }
        token = jwt.encode(payload, secret_key)

        session = await self._get_session()
        async with session.post(
            f"{self.config.upbit.rest_url}/orders",
            headers={"Authorization": f"Bearer {token}"},
            json=params,
        ) as resp:
            data = await resp.json()
            if resp.status in (200, 201):
                return {"success": True, "data": data}
            if resp.status == 429:
                logger.warning("Upbit API Rate limit (429)")
            return {"success": False, "error": f"HTTP {resp.status}: {str(data)[:200]}"}

    # ============================================================
    # Binance Futures REST API
    # ============================================================

    async def _binance_futures_order(self, side: str, qty: float) -> dict:
        """Binance USDM Futures 시장가 주문"""
        api_key = self.config.binance.api_key
        api_secret = self.config.binance.api_secret

        params = {
            "symbol": self.config.trading.binance_futures_symbol,
            "side": "SELL" if side == "sell" else "BUY",
            "type": "MARKET",
            "quantity": f"{qty:.3f}",
            "timestamp": self._binance_timestamp(),
        }

        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            api_secret.encode(), query_string.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature

        session = await self._get_session()
        async with session.post(
            f"{self.config.binance.rest_url}/fapi/v1/order",
            headers={"X-MBX-APIKEY": api_key},
            params=params,
        ) as resp:
            data = await resp.json()
            if resp.status == 200:
                return {"success": True, "data": data}
            if resp.status == 429:
                logger.warning("Binance API Rate limit (429)")
            return {"success": False, "error": f"HTTP {resp.status}: {str(data)[:200]}"}

    # ============================================================
    # 헬퍼
    # ============================================================

    def _parse_leg_result(self, exchange: str, side: str, raw) -> LegResult:
        """API 응답 -> LegResult 변환"""
        if isinstance(raw, Exception):
            return LegResult(
                exchange=exchange, success=False, side=side,
                error=f"{type(raw).__name__}: {raw}",
            )
        if not isinstance(raw, dict):
            return LegResult(
                exchange=exchange, success=False, side=side,
                error=f"unexpected response: {type(raw)}",
            )

        if raw.get("success"):
            data = raw.get("data", {})
            try:
                if exchange == "upbit":
                    return LegResult(
                        exchange=exchange, success=True, side=side,
                        order_id=data.get("uuid", ""),
                        filled_price=float(data.get("price", 0) or data.get("avg_price", 0)),
                        filled_qty=float(data.get("executed_volume", 0)),
                    )
                else:  # binance
                    return LegResult(
                        exchange=exchange, success=True, side=side,
                        order_id=str(data.get("orderId", "")),
                        filled_price=float(data.get("avgPrice", 0)),
                        filled_qty=float(data.get("executedQty", 0)),
                    )
            except (ValueError, TypeError) as e:
                return LegResult(
                    exchange=exchange, success=False, side=side,
                    error=f"응답 파싱 오류: {e}",
                )
        return LegResult(
            exchange=exchange, success=False, side=side,
            error=raw.get("error", "unknown"),
        )

    async def _emergency_unwind(self, result: ExecutionResult):
        """한쪽 레그만 성공한 경우 긴급 청산 (재시도 포함)

        이 함수는 치명적이므로 최대한 성공시켜야 한다.
        config.retry_count와 별도로 최소 3회 재시도한다.
        """
        logger.error(f"PARTIAL FILL -- 긴급 청산 시작: {result.summary()}")
        max_unwind_retries = max(3, self.config.trading.retry_count)

        if result.upbit_leg.success and not result.binance_leg.success:
            # Upbit만 성공 -> Upbit 반대매매
            side = "sell" if result.upbit_leg.side == "buy" else "buy"
            for attempt in range(max_unwind_retries):
                try:
                    unwind = await self._upbit_market_order(side, result.upbit_leg.filled_qty)
                    if unwind.get("success"):
                        logger.info("긴급 Upbit 반대매매 완료")
                        return
                    logger.error(f"긴급 Upbit 반대매매 실패 (시도 {attempt + 1}): {unwind.get('error')}")
                except Exception as e:
                    logger.error(f"긴급 Upbit 반대매매 예외 (시도 {attempt + 1}): {e}")
                if attempt < max_unwind_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
            logger.critical(
                f"긴급 Upbit 반대매매 {max_unwind_retries}회 실패 -- "
                f"수동 청산 필요! qty={result.upbit_leg.filled_qty}"
            )

        elif result.binance_leg.success and not result.upbit_leg.success:
            # Binance만 성공 -> Binance 반대매매
            side = "buy" if result.binance_leg.side == "sell" else "sell"
            for attempt in range(max_unwind_retries):
                try:
                    unwind = await self._binance_futures_order(side, result.binance_leg.filled_qty)
                    if unwind.get("success"):
                        logger.info("긴급 Binance 반대매매 완료")
                        return
                    logger.error(f"긴급 Binance 반대매매 실패 (시도 {attempt + 1}): {unwind.get('error')}")
                except Exception as e:
                    logger.error(f"긴급 Binance 반대매매 예외 (시도 {attempt + 1}): {e}")
                if attempt < max_unwind_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
            logger.critical(
                f"긴급 Binance 반대매매 {max_unwind_retries}회 실패 -- "
                f"수동 청산 필요! qty={result.binance_leg.filled_qty}"
            )

    def _calculate_pnl(self, result: ExecutionResult, snapshot: KPSnapshot) -> float:
        """수익률 계산 (%)

        PnL = (진입KP - 청산KP) - 수수료
        Upbit: 0.05%, Binance Futures: 0.04% (maker/taker 평균)
        양방향이므로 진입+청산 = 4번 수수료
        """
        kp_diff = self.position.entry_kp - snapshot.exit_kp
        fee_pct = (0.05 + 0.04) * 2  # 양쪽 왕복 수수료
        return kp_diff - fee_pct

    def _save_state(self):
        """포지션 상태를 파일에 저장"""
        try:
            self._save_position_fn(self.position.to_dict())
        except Exception as e:
            logger.error(f"포지션 상태 저장 실패 (치명적 아님): {e}")

    async def set_leverage(self) -> bool:
        """Binance Futures 레버리지 설정 (봇 시작 시 1회 호출, 재시도 포함)"""
        leverage = self.config.trading.leverage

        if self.config.trading.dry_run:
            logger.info(f"[DRY RUN] 레버리지 {leverage}x 설정 (시뮬)")
            return True

        api_key = self.config.binance.api_key
        api_secret = self.config.binance.api_secret

        max_retries = self.config.trading.retry_count
        for attempt in range(max_retries + 1):
            try:
                params = {
                    "symbol": self.config.trading.binance_futures_symbol,
                    "leverage": str(leverage),
                    "timestamp": self._binance_timestamp(),
                }

                query_string = urllib.parse.urlencode(params)
                signature = hmac.new(
                    api_secret.encode(), query_string.encode(), hashlib.sha256
                ).hexdigest()
                params["signature"] = signature

                session = await self._get_session()
                async with session.post(
                    f"{self.config.binance.rest_url}/fapi/v1/leverage",
                    headers={"X-MBX-APIKEY": api_key},
                    params=params,
                ) as resp:
                    data = await resp.json()
                    if resp.status == 200:
                        logger.info(f"Binance 레버리지 {leverage}x 설정 완료")
                        return True
                    if resp.status in _NON_RETRYABLE_STATUS:
                        logger.error(f"Binance 레버리지 설정 실패 (재시도 불가): {data}")
                        return False
                    logger.warning(
                        f"Binance 레버리지 설정 실패 "
                        f"(시도 {attempt + 1}/{max_retries + 1}): {data}"
                    )
            except asyncio.TimeoutError:
                logger.warning(f"Binance 레버리지 설정 타임아웃 (시도 {attempt + 1})")
            except Exception as e:
                logger.error(f"Binance 레버리지 설정 오류: {e}")

            if attempt < max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))

        logger.error(f"Binance 레버리지 설정 {max_retries + 1}회 실패")
        return False

    def get_position_info(self) -> dict:
        """현재 포지션 정보"""
        return {
            "side": self.position.side.value,
            "is_open": self.position.is_open,
            "entry_kp": self.position.entry_kp,
            "hold_duration_min": round(self.position.hold_duration_min, 1),
            "upbit_qty": self.position.upbit_qty,
            "binance_qty": self.position.binance_qty,
            "upbit_entry_price": self.position.upbit_entry_price,
            "binance_entry_price": self.position.binance_entry_price,
            "trades_today": self.position.trade_count_today,
        }

"""Kimchirang Data Feeder -- 실시간 WebSocket 데이터 스트림

Upbit (호가/체결), Binance Futures (호가/체결/펀딩비), USD/KRW 환율을
비동기로 동시 수신하여 공유 상태에 저장한다.
"""

import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import websockets

from kimchirang.config import KimchirangConfig

logger = logging.getLogger("kimchirang.feeder")


# ============================================================
# 공유 시장 데이터 (thread-safe 아님 -- asyncio 단일 스레드 가정)
# ============================================================

@dataclass
class OrderBook:
    """호가 스냅샷 (최우선 1호가)"""
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    timestamp: float = 0.0

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_ask - self.best_bid) / self.best_bid * 100
        return 0.0


@dataclass
class TickerData:
    """체결/가격 데이터"""
    last_price: float = 0.0
    volume_24h: float = 0.0
    change_pct_24h: float = 0.0
    timestamp: float = 0.0


@dataclass
class FundingData:
    """Binance Futures 펀딩비"""
    funding_rate: float = 0.0
    next_funding_time: int = 0
    timestamp: float = 0.0


@dataclass
class MarketState:
    """전체 시장 상태 -- 모든 피더가 여기에 기록"""
    # Upbit
    upbit_orderbook: OrderBook = field(default_factory=OrderBook)
    upbit_ticker: TickerData = field(default_factory=TickerData)

    # Binance Futures
    binance_orderbook: OrderBook = field(default_factory=OrderBook)
    binance_ticker: TickerData = field(default_factory=TickerData)
    binance_funding: FundingData = field(default_factory=FundingData)

    # FX Rate
    fx_rate: float = 0.0             # USD/KRW
    fx_updated_at: float = 0.0

    # 연결 상태
    upbit_connected: bool = False
    binance_connected: bool = False
    fx_available: bool = False

    @property
    def is_ready(self) -> bool:
        """모든 데이터가 준비되었는지"""
        return (
            self.upbit_orderbook.best_bid > 0
            and self.binance_orderbook.best_bid > 0
            and self.fx_rate > 0
        )

    @property
    def data_age_sec(self) -> dict:
        """각 데이터의 나이(초)"""
        now = time.time()
        return {
            "upbit_ob": now - self.upbit_orderbook.timestamp if self.upbit_orderbook.timestamp else -1,
            "binance_ob": now - self.binance_orderbook.timestamp if self.binance_orderbook.timestamp else -1,
            "fx": now - self.fx_updated_at if self.fx_updated_at else -1,
        }


# ============================================================
# 데이터 유효성 검증 헬퍼
# ============================================================

def _is_valid_price(price, min_val: float = 0.0, max_val: float = float("inf")) -> bool:
    """가격 데이터 유효성 검증: NaN, Inf, 음수, 극단값 필터링"""
    if price is None:
        return False
    try:
        f = float(price)
        if math.isnan(f) or math.isinf(f):
            return False
        return min_val < f < max_val
    except (TypeError, ValueError):
        return False


# ============================================================
# Upbit WebSocket Feeder
# ============================================================

class UpbitFeeder:
    """Upbit 실시간 호가 + 체결 스트림"""

    MAX_RECONNECT_DELAY = 60
    MAX_RECONNECT_ATTEMPTS = 50  # 약 30분 분량 (backoff 누적)

    def __init__(self, config: KimchirangConfig, state: MarketState):
        self.config = config
        self.state = state
        self._running = False
        self._reconnect_delay = 1.0
        self._reconnect_attempts = 0

    async def run(self):
        """WebSocket 연결 유지 (자동 재연결 + exponential backoff + 최대 재시도)"""
        self._running = True
        while self._running:
            try:
                await self._connect()
            except asyncio.CancelledError:
                raise
            except (websockets.ConnectionClosed, websockets.InvalidURI,
                    websockets.InvalidHandshake) as e:
                logger.warning(f"Upbit WS 연결 끊김 (재연결 예정): {e}")
            except OSError as e:
                logger.error(f"Upbit WS 네트워크 오류: {e}")
            except Exception as e:
                logger.error(f"Upbit WS 예상치 못한 오류: {type(e).__name__}: {e}")
            finally:
                self.state.upbit_connected = False
                self.state.upbit_orderbook = OrderBook()
                self.state.upbit_ticker = TickerData()

            if self._running:
                self._reconnect_attempts += 1
                if self._reconnect_attempts > self.MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        f"Upbit WS 재연결 {self.MAX_RECONNECT_ATTEMPTS}회 초과 "
                        "-- 60초 대기 후 카운터 리셋"
                    )
                    await asyncio.sleep(60)
                    self._reconnect_attempts = 0
                    self._reconnect_delay = 1.0
                    continue
                delay = min(self._reconnect_delay, self.MAX_RECONNECT_DELAY)
                logger.info(
                    f"Upbit WS 재연결 대기 {delay:.0f}초 "
                    f"(시도 {self._reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS})"
                )
                await asyncio.sleep(delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

    async def _connect(self):
        symbol = self.config.trading.upbit_symbol
        subscribe_msg = [
            {"ticket": str(uuid.uuid4())[:8]},
            {"type": "orderbook", "codes": [symbol], "isOnlyRealtime": True},
            {"type": "ticker", "codes": [symbol], "isOnlyRealtime": True},
            {"format": "SIMPLE"},
        ]

        async with websockets.connect(
            self.config.upbit.ws_url,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            await ws.send(json.dumps(subscribe_msg))
            self.state.upbit_connected = True
            self._reconnect_delay = 1.0
            self._reconnect_attempts = 0
            logger.info(f"Upbit WS 연결 완료: {symbol}")

            async for raw in ws:
                try:
                    data = json.loads(raw)
                    self._handle_message(data)
                except json.JSONDecodeError:
                    logger.debug("Upbit WS: JSON 파싱 실패 (무시)")
                    continue
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Upbit WS 메시지 처리 오류 (무시): {e}")
                    continue

    def _handle_message(self, data: dict):
        msg_type = data.get("ty")  # SIMPLE format
        now = time.time()

        if msg_type == "orderbook":
            units = data.get("obu", [])
            if units:
                top = units[0]
                bid = float(top.get("bp", 0))
                ask = float(top.get("ap", 0))
                # 유효성 검증: KRW 가격은 양수이고 합리적 범위여야 함
                if not _is_valid_price(bid, min_val=1000) or not _is_valid_price(ask, min_val=1000):
                    logger.warning(f"Upbit 호가 비정상 값 무시: bid={bid}, ask={ask}")
                    return
                if ask < bid:
                    logger.warning(f"Upbit 호가 역전 무시: bid={bid} > ask={ask}")
                    return
                ob = self.state.upbit_orderbook
                ob.best_bid = bid
                ob.best_ask = ask
                ob.bid_qty = max(0.0, float(top.get("bs", 0)))
                ob.ask_qty = max(0.0, float(top.get("as", 0)))
                ob.timestamp = now

        elif msg_type == "ticker":
            price = float(data.get("tp", 0))
            if _is_valid_price(price, min_val=1000):
                tk = self.state.upbit_ticker
                tk.last_price = price
                tk.volume_24h = max(0.0, float(data.get("atv24h", 0)))
                tk.change_pct_24h = float(data.get("scr", 0)) * 100
                tk.timestamp = now

    def stop(self):
        self._running = False


# ============================================================
# Binance Futures WebSocket Feeder
# ============================================================

class BinanceFeeder:
    """Binance USDM Futures 실시간 호가 + 체결 + 마크 가격 스트림"""

    MAX_RECONNECT_DELAY = 60
    MAX_RECONNECT_ATTEMPTS = 50

    def __init__(self, config: KimchirangConfig, state: MarketState):
        self.config = config
        self.state = state
        self._running = False
        self._reconnect_delay = 1.0
        self._reconnect_attempts = 0

    async def run(self):
        self._running = True
        while self._running:
            try:
                await self._connect()
            except asyncio.CancelledError:
                raise
            except (websockets.ConnectionClosed, websockets.InvalidURI,
                    websockets.InvalidHandshake) as e:
                logger.warning(f"Binance WS 연결 끊김 (재연결 예정): {e}")
            except OSError as e:
                logger.error(f"Binance WS 네트워크 오류: {e}")
            except Exception as e:
                logger.error(f"Binance WS 예상치 못한 오류: {type(e).__name__}: {e}")
            finally:
                self.state.binance_connected = False
                self.state.binance_orderbook = OrderBook()
                self.state.binance_ticker = TickerData()
                self.state.binance_funding = FundingData()

            if self._running:
                self._reconnect_attempts += 1
                if self._reconnect_attempts > self.MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        f"Binance WS 재연결 {self.MAX_RECONNECT_ATTEMPTS}회 초과 "
                        "-- 60초 대기 후 카운터 리셋"
                    )
                    await asyncio.sleep(60)
                    self._reconnect_attempts = 0
                    self._reconnect_delay = 1.0
                    continue
                delay = min(self._reconnect_delay, self.MAX_RECONNECT_DELAY)
                logger.info(
                    f"Binance WS 재연결 대기 {delay:.0f}초 "
                    f"(시도 {self._reconnect_attempts}/{self.MAX_RECONNECT_ATTEMPTS})"
                )
                await asyncio.sleep(delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

    async def _connect(self):
        symbol_lower = self.config.trading.binance_futures_symbol.lower()
        # Combined stream: bookTicker(최우선호가) + ticker + markPrice
        streams = f"{symbol_lower}@bookTicker/{symbol_lower}@ticker/{symbol_lower}@markPrice"
        base_url = self.config.binance.ws_url.replace("/ws", "/stream", 1)
        url = f"{base_url}?streams={streams}"

        async with websockets.connect(
            url,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self.state.binance_connected = True
            self._reconnect_delay = 1.0
            self._reconnect_attempts = 0
            logger.info(f"Binance Futures WS 연결 완료: {symbol_lower}")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    data = msg.get("data", msg)
                    event = data.get("e", "")
                    self._handle_message(event, data)
                except json.JSONDecodeError:
                    logger.debug("Binance WS: JSON 파싱 실패 (무시)")
                    continue
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Binance WS 메시지 처리 오류 (무시): {e}")
                    continue

    def _handle_message(self, event: str, data: dict):
        now = time.time()

        if event == "bookTicker":
            bid = float(data.get("b", 0))
            ask = float(data.get("a", 0))
            # USDT 가격 유효성 검증
            if not _is_valid_price(bid, min_val=100) or not _is_valid_price(ask, min_val=100):
                logger.warning(f"Binance 호가 비정상 값 무시: bid={bid}, ask={ask}")
                return
            if ask < bid:
                logger.warning(f"Binance 호가 역전 무시: bid={bid} > ask={ask}")
                return
            ob = self.state.binance_orderbook
            ob.best_bid = bid
            ob.best_ask = ask
            ob.bid_qty = max(0.0, float(data.get("B", 0)))
            ob.ask_qty = max(0.0, float(data.get("A", 0)))
            ob.timestamp = now

        elif event == "24hrTicker":
            price = float(data.get("c", 0))
            if _is_valid_price(price, min_val=100):
                tk = self.state.binance_ticker
                tk.last_price = price
                tk.volume_24h = max(0.0, float(data.get("v", 0)))
                tk.change_pct_24h = float(data.get("P", 0))
                tk.timestamp = now

        elif event == "markPriceUpdate":
            funding_rate = float(data.get("r", 0))
            # 펀딩비 합리성 체크: 보통 -0.75% ~ +0.75% 범위
            if abs(funding_rate) > 0.01:
                logger.warning(f"Binance 펀딩비 이상값 무시: {funding_rate}")
                return
            fd = self.state.binance_funding
            fd.funding_rate = funding_rate
            fd.next_funding_time = int(data.get("T", 0))
            fd.timestamp = now

    def stop(self):
        self._running = False


# ============================================================
# FX Rate Fetcher (USD/KRW)
# ============================================================

class FXFeeder:
    """USD/KRW 환율 주기적 갱신"""

    # 환율 합리성 범위 (KRW/USD)
    FX_MIN = 900.0
    FX_MAX = 2000.0
    # 기존 값 유지 최대 시간 (초)
    FX_STALE_TIMEOUT = 600
    # 연속 실패 허용 횟수
    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, config: KimchirangConfig, state: MarketState):
        self.config = config
        self.state = state
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._consecutive_failures = 0

    async def run(self):
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        )
        try:
            while self._running:
                try:
                    rate = await self._fetch_rate()
                    if rate and self.FX_MIN < rate < self.FX_MAX:
                        # 기존 값 대비 급변 체크 (5% 이상 변동은 의심)
                        if (self.state.fx_rate > 0
                                and abs(rate - self.state.fx_rate) / self.state.fx_rate > 0.05):
                            logger.warning(
                                f"FX Rate 급변 감지: {self.state.fx_rate:.2f} -> {rate:.2f} (무시)"
                            )
                        else:
                            self.state.fx_rate = rate
                            self.state.fx_updated_at = time.time()
                            self.state.fx_available = True
                            self._consecutive_failures = 0
                            logger.info(f"USD/KRW 환율 갱신: {rate:,.2f}")
                    else:
                        self._consecutive_failures += 1
                        logger.warning(
                            f"비정상 환율: {rate} "
                            f"(연속실패 {self._consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES})"
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self._consecutive_failures += 1
                    logger.error(
                        f"FX Rate 갱신 실패: {e} "
                        f"(연속실패 {self._consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES})"
                    )

                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"FX Rate {self.MAX_CONSECUTIVE_FAILURES}회 연속 실패 "
                        "-- fx_available=False 설정"
                    )
                    if time.time() - self.state.fx_updated_at > self.FX_STALE_TIMEOUT:
                        self.state.fx_available = False

                await asyncio.sleep(self.config.trading.fx_update_interval_sec)
        finally:
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None

    async def _fetch_rate(self) -> Optional[float]:
        """여러 소스에서 USD/KRW 환율 조회"""
        session = self._session
        if session is None or session.closed:
            return None

        # 소스 1: exchangerate-api (무료, 안정적)
        try:
            async with session.get(
                "https://open.er-api.com/v6/latest/USD",
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    rate = data.get("rates", {}).get("KRW")
                    if rate:
                        return float(rate)
                elif resp.status == 429:
                    logger.warning("FX 소스1 Rate limit (429)")
                else:
                    logger.debug(f"FX 소스1 HTTP {resp.status}")
        except asyncio.TimeoutError:
            logger.debug("FX 소스1 타임아웃")
        except Exception as e:
            logger.debug(f"FX 소스1 실패: {e}")

        # 소스 2: Upbit USDT/KRW (크립토 기준, 근사)
        try:
            async with session.get(
                "https://api.upbit.com/v1/ticker?markets=KRW-USDT",
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and isinstance(data, list):
                        price = float(data[0].get("trade_price", 0))
                        if price > 0:
                            return price
        except asyncio.TimeoutError:
            logger.debug("FX 소스2 타임아웃")
        except Exception as e:
            logger.debug(f"FX 소스2 실패: {e}")

        # 소스 3: 기존 값 유지 (stale timeout 이내)
        if self.state.fx_rate > 0 and time.time() - self.state.fx_updated_at < self.FX_STALE_TIMEOUT:
            logger.warning("FX Rate 갱신 실패 -- 기존 값 유지")
            return self.state.fx_rate

        return None

    def stop(self):
        self._running = False


# ============================================================
# 통합 DataFeeder
# ============================================================

class DataFeeder:
    """모든 피더를 관리하는 통합 클래스"""

    def __init__(self, config: KimchirangConfig):
        self.config = config
        self.state = MarketState()
        self._upbit = UpbitFeeder(config, self.state)
        self._binance = BinanceFeeder(config, self.state)
        self._fx = FXFeeder(config, self.state)
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """모든 피더 시작"""
        logger.info("DataFeeder 시작...")
        self._tasks = [
            asyncio.create_task(self._upbit.run(), name="upbit_ws"),
            asyncio.create_task(self._binance.run(), name="binance_ws"),
            asyncio.create_task(self._fx.run(), name="fx_rate"),
        ]
        # 피더 태스크 비정상 종료 감지
        for task in self._tasks:
            task.add_done_callback(self._on_task_done)
        logger.info("3개 피더 태스크 시작 완료")

    def _on_task_done(self, task: asyncio.Task):
        """피더 태스크가 예상치 못하게 종료되면 로깅"""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"피더 태스크 '{task.get_name()}' 비정상 종료: {exc}")

    async def stop(self):
        """모든 피더 정지"""
        self._upbit.stop()
        self._binance.stop()
        self._fx.stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("DataFeeder 정지 완료")

    async def wait_ready(self, timeout: float = 30) -> bool:
        """모든 데이터가 준비될 때까지 대기"""
        start = time.time()
        while time.time() - start < timeout:
            if self.state.is_ready:
                logger.info(
                    f"데이터 준비 완료 ({time.time() - start:.1f}초) -- "
                    f"Upbit: {self.state.upbit_orderbook.best_bid:,.0f} KRW, "
                    f"Binance: {self.state.binance_orderbook.best_bid:,.2f} USDT, "
                    f"FX: {self.state.fx_rate:,.2f} KRW/USD"
                )
                return True
            # 피더 태스크가 전부 죽었으면 빠른 실패
            if all(t.done() for t in self._tasks):
                logger.error("모든 피더 태스크 종료됨 -- 데이터 준비 불가")
                return False
            await asyncio.sleep(0.5)
        logger.error(f"데이터 준비 타임아웃 ({timeout}초)")
        return False

    def get_status(self) -> dict:
        """현재 상태 요약"""
        return {
            "ready": self.state.is_ready,
            "upbit_connected": self.state.upbit_connected,
            "binance_connected": self.state.binance_connected,
            "fx_available": self.state.fx_available,
            "fx_rate": self.state.fx_rate,
            "data_age": self.state.data_age_sec,
            "upbit_bid": self.state.upbit_orderbook.best_bid,
            "upbit_ask": self.state.upbit_orderbook.best_ask,
            "binance_bid": self.state.binance_orderbook.best_bid,
            "binance_ask": self.state.binance_orderbook.best_ask,
        }

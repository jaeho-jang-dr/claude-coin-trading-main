"""Kimchirang Data Feeder -- 실시간 WebSocket 데이터 스트림

Upbit (호가/체결), Binance Futures (호가/체결/펀딩비), USD/KRW 환율을
비동기로 동시 수신하여 공유 상태에 저장한다.
"""

import asyncio
import json
import logging
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
        if self.best_bid > 0:
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
# Upbit WebSocket Feeder
# ============================================================

class UpbitFeeder:
    """Upbit 실시간 호가 + 체결 스트림"""

    def __init__(self, config: KimchirangConfig, state: MarketState):
        self.config = config
        self.state = state
        self._running = False
        self._reconnect_delay = 1

    async def run(self):
        """WebSocket 연결 유지 (자동 재연결)"""
        self._running = True
        while self._running:
            try:
                await self._connect()
            except Exception as e:
                logger.error(f"Upbit WS 오류: {e}")
                self.state.upbit_connected = False
                await asyncio.sleep(min(self._reconnect_delay, 30))
                self._reconnect_delay *= 2

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
            self._reconnect_delay = 1
            logger.info(f"Upbit WS 연결 완료: {symbol}")

            async for raw in ws:
                try:
                    data = json.loads(raw)
                    self._handle_message(data)
                except json.JSONDecodeError:
                    continue

    def _handle_message(self, data: dict):
        msg_type = data.get("ty")  # SIMPLE format
        now = time.time()

        if msg_type == "orderbook":
            units = data.get("obu", [])
            if units:
                top = units[0]
                self.state.upbit_orderbook = OrderBook(
                    best_bid=float(top.get("bp", 0)),
                    best_ask=float(top.get("ap", 0)),
                    bid_qty=float(top.get("bs", 0)),
                    ask_qty=float(top.get("as", 0)),
                    timestamp=now,
                )

        elif msg_type == "ticker":
            self.state.upbit_ticker = TickerData(
                last_price=float(data.get("tp", 0)),
                volume_24h=float(data.get("atv24h", 0)),
                change_pct_24h=float(data.get("scr", 0)) * 100,
                timestamp=now,
            )

    def stop(self):
        self._running = False


# ============================================================
# Binance Futures WebSocket Feeder
# ============================================================

class BinanceFeeder:
    """Binance USDM Futures 실시간 호가 + 체결 + 마크 가격 스트림"""

    def __init__(self, config: KimchirangConfig, state: MarketState):
        self.config = config
        self.state = state
        self._running = False
        self._reconnect_delay = 1

    async def run(self):
        self._running = True
        while self._running:
            try:
                await self._connect()
            except Exception as e:
                logger.error(f"Binance WS 오류: {e}")
                self.state.binance_connected = False
                await asyncio.sleep(min(self._reconnect_delay, 30))
                self._reconnect_delay *= 2

    async def _connect(self):
        symbol_lower = self.config.trading.binance_futures_symbol.lower()
        # Combined stream: bookTicker(최우선호가) + ticker + markPrice
        streams = f"{symbol_lower}@bookTicker/{symbol_lower}@ticker/{symbol_lower}@markPrice"
        url = f"{self.config.binance.ws_url}/{streams}"

        async with websockets.connect(
            url,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self.state.binance_connected = True
            self._reconnect_delay = 1
            logger.info(f"Binance Futures WS 연결 완료: {symbol_lower}")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    # Combined stream format: {"stream": "...", "data": {...}}
                    data = msg.get("data", msg)
                    event = data.get("e", "")
                    self._handle_message(event, data)
                except json.JSONDecodeError:
                    continue

    def _handle_message(self, event: str, data: dict):
        now = time.time()

        if event == "bookTicker":
            self.state.binance_orderbook = OrderBook(
                best_bid=float(data.get("b", 0)),
                best_ask=float(data.get("a", 0)),
                bid_qty=float(data.get("B", 0)),
                ask_qty=float(data.get("A", 0)),
                timestamp=now,
            )

        elif event == "24hrTicker":
            self.state.binance_ticker = TickerData(
                last_price=float(data.get("c", 0)),
                volume_24h=float(data.get("v", 0)),
                change_pct_24h=float(data.get("P", 0)),
                timestamp=now,
            )

        elif event == "markPriceUpdate":
            self.state.binance_funding = FundingData(
                funding_rate=float(data.get("r", 0)),
                next_funding_time=int(data.get("T", 0)),
                timestamp=now,
            )

    def stop(self):
        self._running = False


# ============================================================
# FX Rate Fetcher (USD/KRW)
# ============================================================

class FXFeeder:
    """USD/KRW 환율 주기적 갱신"""

    # 무료 환율 API 우선순위
    FX_SOURCES = [
        # 한국수출입은행 API (공식, 영업일만)
        "https://www.koreaexim.go.kr/site/program/financial/exchangeJSON?authkey={authkey}&data=AP01&searchdate={date}",
        # 대체: exchangerate-api.com
        "https://open.er-api.com/v6/latest/USD",
        # 대체: 고정 fallback (비상용)
    ]

    def __init__(self, config: KimchirangConfig, state: MarketState):
        self.config = config
        self.state = state
        self._running = False

    async def run(self):
        self._running = True
        while self._running:
            try:
                rate = await self._fetch_rate()
                if rate and rate > 1000:  # 합리성 체크
                    self.state.fx_rate = rate
                    self.state.fx_updated_at = time.time()
                    self.state.fx_available = True
                    logger.info(f"USD/KRW 환율 갱신: {rate:,.2f}")
                else:
                    logger.warning(f"비정상 환율: {rate}")
            except Exception as e:
                logger.error(f"FX Rate 갱신 실패: {e}")

            await asyncio.sleep(self.config.trading.fx_update_interval_sec)

    async def _fetch_rate(self) -> Optional[float]:
        """여러 소스에서 USD/KRW 환율 조회"""
        # 소스 1: exchangerate-api (무료, 안정적)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://open.er-api.com/v6/latest/USD",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rate = data.get("rates", {}).get("KRW")
                        if rate:
                            return float(rate)
        except Exception as e:
            logger.debug(f"FX 소스1 실패: {e}")

        # 소스 2: Upbit USDT/KRW (크립토 기준, 근사)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.upbit.com/v1/ticker?markets=KRW-USDT",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and isinstance(data, list):
                            return float(data[0].get("trade_price", 0))
        except Exception as e:
            logger.debug(f"FX 소스2 실패: {e}")

        # 소스 3: 기존 값 유지 (10분 이내)
        if self.state.fx_rate > 0 and time.time() - self.state.fx_updated_at < 600:
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
        logger.info("3개 피더 태스크 시작 완료")

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

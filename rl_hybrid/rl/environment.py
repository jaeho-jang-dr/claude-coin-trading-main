"""BitcoinTradingEnv — Gymnasium 커스텀 환경

Upbit 히스토리컬 데이터 기반 비트코인 트레이딩 시뮬레이터.
연속 행동 공간(-1 ~ +1)에서 포트폴리오 비중을 조절한다.

Action:
    -1.0 = 전량 매도 (BTC → KRW 100%)
     0.0 = 관망 (포지션 유지)
    +1.0 = 전량 매수 (KRW → BTC 100%)

Observation:
    42차원 정규화 벡터 (StateEncoder 출력)

Reward:
    샤프 지수 기반 위험 조정 수익률 (RewardCalculator)
"""

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM
from rl_hybrid.rl.reward import RewardCalculator, TRANSACTION_COST
from rl_hybrid.rl.data_loader import HistoricalDataLoader

logger = logging.getLogger("rl.environment")


class BitcoinTradingEnv(gym.Env):
    """비트코인 트레이딩 Gymnasium 환경"""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        candles: list[dict] = None,
        initial_balance: float = 10_000_000,  # 1000만원
        max_steps: int = None,
        lookback: int = 24,  # 과거 N봉 참조
        render_mode: str = None,
        external_signals: list[dict] | None = None,
    ):
        """
        Args:
            candles: 지표 계산 완료된 캔들 데이터 (data_loader.compute_indicators 출력)
            initial_balance: 초기 KRW 잔고
            max_steps: 에피소드 최대 스텝 (None이면 전체 데이터 사용)
            lookback: 관측에 포함할 과거 봉 수
            render_mode: "human" 또는 "ansi"
            external_signals: 캔들과 동일 길이로 정렬된 외부 시그널 리스트.
                None이면 기본 상수값 사용 (기존 동작).
        """
        super().__init__()

        # 데이터 로드
        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=180, interval="1h")
            candles = loader.compute_indicators(raw)

        self.candles = candles
        self.external_signals = external_signals
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.render_mode = render_mode

        # 에피소드 범위: lookback 이후부터
        self.start_idx = lookback
        self.end_idx = len(candles) - 1
        if max_steps:
            self.end_idx = min(self.start_idx + max_steps, self.end_idx)
        # Guard: very short datasets (< lookback candles)
        if self.end_idx <= self.start_idx:
            self.end_idx = len(candles) - 1
            self.start_idx = min(self.start_idx, self.end_idx - 1)
        self.start_idx = max(0, self.start_idx)

        # 관측 공간: 42차원 (현재 상태) + lookback * 5 (과거 OHLCV 압축)
        # 간소화: 42차원만 사용 (과거 정보는 지표에 이미 반영)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )

        # 행동 공간: 연속 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 내부 구성요소
        self.encoder = StateEncoder()
        self.reward_calc = RewardCalculator()

        # 상태 변수 (reset에서 초기화)
        self.current_step = 0
        self.krw_balance = 0.0
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.total_value_history = []
        self.action_history = []
        self.trade_count = 0

        # 외부 데이터 캐시 (signals가 없으면 매 스텝 동일한 dict 반환)
        self._cached_external_data: dict | None = None
        self._cached_external_step: int = -1

    def reset(self, seed=None, options=None):
        """환경 리셋"""
        super().reset(seed=seed)

        # 랜덤 시작점 (학습 다양성)
        if options and options.get("start_idx"):
            self.current_step = options["start_idx"]
        else:
            max_start = max(self.start_idx, self.end_idx - 500)
            self.current_step = self.np_random.integers(self.start_idx, max_start + 1)

        self.krw_balance = self.initial_balance
        self.btc_balance = 0.0
        self.prev_action = 0.0
        self.total_value_history = [self.initial_balance]
        self.action_history = []
        self.trade_count = 0
        self._cached_external_data = None
        self._cached_external_step = -1

        self.reward_calc.reset(self.initial_balance)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        """1스텝 실행

        Args:
            action: np.ndarray shape (1,), 값 [-1, 1]
                -1: 전량 매도, 0: 관망, +1: 전량 매수

        Returns:
            observation, reward, terminated, truncated, info
        """
        action_val = float(np.clip(action[0], -1, 1))
        candle = self.candles[self.current_step]
        price = candle["close"]

        # 이전 포트폴리오 가치
        prev_value = self._portfolio_value(price)

        # 행동 실행
        self._execute_action(action_val, price)

        # 다음 스텝
        self.current_step += 1
        next_candle = self.candles[self.current_step]
        next_price = next_candle["close"]

        # 현재 포트폴리오 가치
        curr_value = self._portfolio_value(next_price)
        self.total_value_history.append(curr_value)
        self.action_history.append(action_val)

        # 보상 계산
        reward_info = self.reward_calc.calculate(
            prev_portfolio_value=prev_value,
            curr_portfolio_value=curr_value,
            action=action_val,
            prev_action=self.prev_action,
            step=self.current_step,
        )

        self.prev_action = action_val

        # 종료 조건
        terminated = False
        truncated = self.current_step >= self.end_idx

        # 파산 체크 (가치 90% 이하)
        if curr_value < self.initial_balance * 0.1:
            terminated = True
            reward_info["reward"] -= 1.0  # 파산 페널티

        obs = self._get_observation()
        info = self._get_info()
        info["reward_components"] = reward_info["components"]

        return obs, reward_info["reward"], terminated, truncated, info

    def _execute_action(self, action: float, price: float):
        """행동 실행 — 포지션 비중 조절

        action > 0: 매수 (KRW → BTC)
        action < 0: 매도 (BTC → KRW)
        |action| = 비중 변경 비율
        """
        total_value = self._portfolio_value(price)
        target_btc_ratio = (action + 1) / 2  # [-1,1] → [0,1]

        current_btc_value = self.btc_balance * price
        current_btc_ratio = current_btc_value / total_value if total_value > 0 else 0
        target_btc_value = total_value * target_btc_ratio

        diff = target_btc_value - current_btc_value

        if diff > 0 and self.krw_balance > 0:
            # 매수
            buy_amount = min(diff, self.krw_balance)
            cost = buy_amount * TRANSACTION_COST
            actual_buy = buy_amount - cost
            btc_bought = actual_buy / price
            self.krw_balance -= buy_amount
            self.btc_balance += btc_bought
            if buy_amount > total_value * 0.01:
                self.trade_count += 1

        elif diff < 0 and self.btc_balance > 0:
            # 매도
            sell_value = min(-diff, current_btc_value)
            btc_sold = sell_value / price
            btc_sold = min(btc_sold, self.btc_balance)
            proceeds = btc_sold * price * (1 - TRANSACTION_COST)
            self.btc_balance -= btc_sold
            self.krw_balance += proceeds
            if sell_value > total_value * 0.01:
                self.trade_count += 1

    def _portfolio_value(self, price: float) -> float:
        """현재 포트폴리오 가치 (KRW)"""
        return self.krw_balance + self.btc_balance * price

    # 재사용 가능한 고정 dict 템플릿 (매 스텝 동일한 부분)
    _STATIC_ORDERBOOK = {"ratio": 1.0}
    _STATIC_TRADE_PRESSURE = {"buy_volume": 1, "sell_volume": 1}
    _STATIC_ETH_BTC = {"eth_btc_z_score": 0}

    def _get_observation(self) -> np.ndarray:
        """현재 관측 벡터 생성"""
        candle = self.candles[self.current_step]
        price = candle["close"]

        # 시장 데이터 → StateEncoder 호환 형식
        # 내부 dict를 매번 새로 만들되, 고정 부분은 클래스 상수 참조
        market_data = {
            "current_price": price,
            "change_rate_24h": candle.get("change_rate", 0),
            "indicators": {
                "sma_20": candle.get("sma_20", price),
                "sma_50": candle.get("sma_50", price),
                "rsi_14": candle.get("rsi_14", 50),
                "macd": {
                    "macd": candle.get("macd", 0),
                    "signal": candle.get("macd_signal", 0),
                    "histogram": candle.get("macd_histogram", 0),
                },
                "bollinger": {
                    "upper": candle.get("boll_upper", price),
                    "middle": candle.get("boll_middle", price),
                    "lower": candle.get("boll_lower", price),
                },
                "stochastic": {
                    "k": candle.get("stoch_k", 50),
                    "d": candle.get("stoch_d", 50),
                },
                "adx": {
                    "adx": candle.get("adx", 25),
                    "plus_di": candle.get("adx_plus_di", 20),
                    "minus_di": candle.get("adx_minus_di", 20),
                },
                "atr": candle.get("atr", 0),
            },
            "indicators_4h": {"rsi_14": candle.get("rsi_14", 50)},
            "orderbook": self._STATIC_ORDERBOOK,
            "trade_pressure": self._STATIC_TRADE_PRESSURE,
            "eth_btc_analysis": self._STATIC_ETH_BTC,
        }

        # 외부 데이터: 실제 시그널이 있으면 사용, 없으면 기본 상수값 (캐시됨)
        external_data = self._build_external_data()

        # 포트폴리오
        total_value = self._portfolio_value(price)
        btc_eval = self.btc_balance * price
        if self.btc_balance > 0:
            holdings = [{
                "currency": "BTC",
                "balance": self.btc_balance,
                "avg_buy_price": price,
                "eval_amount": btc_eval,
                "profit_loss_pct": 0,
            }]
        else:
            holdings = []

        portfolio = {
            "krw_balance": self.krw_balance,
            "holdings": holdings,
            "total_eval": total_value,
        }

        # 에이전트 상태 — 재사용 가능한 dict (trade_count만 변동)
        agent_state = {
            "danger_score": 30,
            "opportunity_score": 30,
            "cascade_risk": 20,
            "consecutive_losses": 0,
            "hours_since_last_trade": 24,
            "daily_trade_count": self.trade_count,
        }

        return self.encoder.encode(market_data, external_data, portfolio, agent_state)

    # === 감성 텍스트 → 숫자 변환 맵 ===
    _SENTIMENT_MAP = {
        "very_positive": 80,
        "positive": 50,
        "slightly_positive": 25,
        "neutral": 0,
        "slightly_negative": -25,
        "negative": -50,
        "very_negative": -80,
    }

    def _build_external_data(self) -> dict:
        """외부 데이터 dict 생성 — StateEncoder 호환 형식

        self.external_signals가 있고 현재 스텝에 데이터가 있으면
        실제 값을 사용하고, 없으면 기본 상수값을 반환한다.

        Caching: 외부 시그널이 없을 때는 매 스텝 동일한 dict이므로 캐시.
        시그널이 있을 때는 스텝별로 다르므로 스텝 기준 캐시.
        """
        # 캐시 히트 확인
        if self._cached_external_data is not None:
            if self.external_signals is None:
                # 시그널 없으면 항상 동일 → 캐시 반환
                return self._cached_external_data
            if self._cached_external_step == self.current_step:
                return self._cached_external_data

        # 기본값
        fgi = 50
        news_score = 0
        whale = 0
        funding = 0.0
        ls_ratio = 1.0
        kimchi = 0.0
        macro = 0
        fusion = 0
        nvt = 100.0
        # eth_btc_score는 외부 시그널에 있지만 StateEncoder는
        # market_data.eth_btc_analysis에서 읽으므로 여기선 사용 안 함

        if (
            self.external_signals is not None
            and self.current_step < len(self.external_signals)
        ):
            sig = self.external_signals[self.current_step]

            fgi = sig.get("fgi_value", 50) or 50
            whale = sig.get("whale_score", 0) or 0
            funding = sig.get("funding_rate", 0.0) or 0.0
            ls_ratio = sig.get("long_short_ratio", 1.0) or 1.0
            kimchi = sig.get("kimchi_premium_pct", 0.0) or 0.0
            macro = sig.get("macro_score", 0) or 0
            fusion = sig.get("fusion_score", 0) or 0
            nvt = sig.get("nvt_signal", 100.0) or 100.0

            # news_sentiment: 텍스트 → 숫자 변환
            raw_sentiment = sig.get("news_sentiment", "neutral")
            if isinstance(raw_sentiment, (int, float)):
                news_score = float(raw_sentiment)
            elif isinstance(raw_sentiment, str):
                news_score = self._SENTIMENT_MAP.get(
                    raw_sentiment.lower().strip(), 0
                )
            else:
                news_score = 0

        result = {
            "sources": {
                "fear_greed": {"current": {"value": float(fgi)}},
                "news_sentiment": {"sentiment_score": float(news_score)},
                "whale_tracker": {"whale_score": {"score": float(whale)}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": float(funding)},
                    "top_trader_long_short": {"current_ratio": float(ls_ratio)},
                    "kimchi_premium": {"premium_pct": float(kimchi)},
                },
                "macro": {"analysis": {"macro_score": float(macro)}},
                "ai_signal": {"ai_composite_signal": {"score": 0}},
                "coinmarketcap": {"btc_dominance": 50},
            },
            "external_signal": {"total_score": float(fusion)},
            "nvt_signal": float(nvt),
        }

        self._cached_external_data = result
        self._cached_external_step = self.current_step
        return result

    def _get_info(self) -> dict:
        """디버깅/로깅용 info dict"""
        candle = self.candles[self.current_step]
        price = candle["close"]
        total_value = self._portfolio_value(price)

        return {
            "step": self.current_step,
            "timestamp": candle.get("timestamp", ""),
            "price": price,
            "portfolio_value": total_value,
            "krw_balance": self.krw_balance,
            "btc_balance": self.btc_balance,
            "btc_ratio": (self.btc_balance * price / total_value) if total_value > 0 else 0,
            "return_pct": (total_value - self.initial_balance) / self.initial_balance * 100,
            "trade_count": self.trade_count,
        }

    def render(self):
        """현재 상태 출력"""
        info = self._get_info()
        if self.render_mode == "ansi":
            return (
                f"[Step {info['step']}] {info['timestamp']} | "
                f"BTC: {info['price']:,.0f}원 | "
                f"포트폴리오: {info['portfolio_value']:,.0f}원 "
                f"({info['return_pct']:+.2f}%) | "
                f"BTC비중: {info['btc_ratio']:.1%} | "
                f"거래: {info['trade_count']}회"
            )
        elif self.render_mode == "human":
            # Build state string directly (avoid recursion)
            price = self.candles[self.current_step]["close"]
            portfolio_value = self._portfolio_value(price)
            state = (f"Step {self.current_step} | "
                     f"Balance: {self.krw_balance:,.0f} | "
                     f"BTC: {self.btc_balance:.6f} | "
                     f"Value: {portfolio_value:,.0f}")
            print(state)
            return state

    def get_episode_stats(self) -> dict:
        """에피소드 통계"""
        final_value = self.total_value_history[-1]
        stats = self.reward_calc.get_episode_stats(final_value, self.initial_balance)
        stats["trade_count"] = self.trade_count
        stats["final_value"] = final_value
        stats["steps"] = len(self.total_value_history) - 1
        return stats


class BitcoinTradingEnvWithLLM(BitcoinTradingEnv):
    """LLM 임베딩을 관측에 포함하는 확장 환경

    Phase 3에서 Gemini 분석 벡터를 관측 공간에 concat한다.
    """

    def __init__(self, llm_embedding_dim: int = 3072, **kwargs):
        super().__init__(**kwargs)
        self.llm_embedding_dim = llm_embedding_dim
        self.llm_embedding = np.zeros(llm_embedding_dim, dtype=np.float32)

        # 관측 공간 확장 (LLM 임베딩은 음수 가능)
        total_dim = OBSERVATION_DIM + llm_embedding_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def set_llm_embedding(self, embedding: np.ndarray):
        """외부에서 LLM 임베딩 주입"""
        self.llm_embedding = embedding.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        base_obs = super()._get_observation()
        return np.concatenate([base_obs, self.llm_embedding])

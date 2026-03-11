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
    ):
        """
        Args:
            candles: 지표 계산 완료된 캔들 데이터 (data_loader.compute_indicators 출력)
            initial_balance: 초기 KRW 잔고
            max_steps: 에피소드 최대 스텝 (None이면 전체 데이터 사용)
            lookback: 관측에 포함할 과거 봉 수
            render_mode: "human" 또는 "ansi"
        """
        super().__init__()

        # 데이터 로드
        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=180, interval="1h")
            candles = loader.compute_indicators(raw)

        self.candles = candles
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.render_mode = render_mode

        # 에피소드 범위: lookback 이후부터
        self.start_idx = lookback
        self.end_idx = len(candles) - 1
        if max_steps:
            self.end_idx = min(self.start_idx + max_steps, self.end_idx)

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
        self.avg_buy_price = 0.0  # 실제 평균 매수가 추적
        self.prev_action = 0.0
        self.total_value_history = []
        self.action_history = []
        self.trade_count = 0

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
        self.avg_buy_price = 0.0
        self.prev_action = 0.0
        self.total_value_history = [self.initial_balance]
        self.action_history = []
        self.trade_count = 0

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
            # 평균 매수가 갱신 (가중 평균)
            if self.btc_balance + btc_bought > 0:
                total_cost = self.avg_buy_price * self.btc_balance + price * btc_bought
                self.avg_buy_price = total_cost / (self.btc_balance + btc_bought)
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

    def _get_observation(self) -> np.ndarray:
        """현재 관측 벡터 생성"""
        candle = self.candles[self.current_step]
        price = candle["close"]

        # 시장 데이터 → StateEncoder 호환 형식
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
            "orderbook": {"ratio": 1.0},
            "trade_pressure": {"buy_volume": 1, "sell_volume": 1},
            "eth_btc_analysis": {"eth_btc_z_score": 0},
        }

        # 외부 데이터 (시뮬레이션: 가격 기반 동적 생성 + 노이즈)
        # 가격 변동률로 시뮬레이션 감성 데이터 생성 (일정값 대신)
        change_rate = candle.get("change_rate", 0)
        rsi = candle.get("rsi_14", 50)
        rng = self.np_random

        # RSI/가격변동에 연동된 시뮬레이션 FGI (50 ± 노이즈)
        sim_fgi = np.clip(50 + change_rate * 200 + rng.normal(0, 8), 5, 95)
        # 뉴스 감성: 가격 변동 방향 + 노이즈
        sim_news = np.clip(change_rate * 300 + rng.normal(0, 10), -50, 50)
        # 펀딩 레이트: RSI 기반
        sim_funding = np.clip((rsi - 50) * 0.002 + rng.normal(0, 0.005), -0.05, 0.05)

        external_data = {
            "sources": {
                "fear_greed": {"current": {"value": float(sim_fgi)}},
                "news_sentiment": {"sentiment_score": float(sim_news)},
                "whale_tracker": {"whale_score": {"score": float(rng.normal(0, 5))}},
                "binance_sentiment": {
                    "funding_rate": {"current_rate": float(sim_funding)},
                    "top_trader_long_short": {"current_ratio": float(np.clip(1.0 + rng.normal(0, 0.2), 0.5, 2.0))},
                    "kimchi_premium": {"premium_pct": float(rng.normal(1.5, 0.8))},
                },
                "macro": {"analysis": {"macro_score": float(rng.normal(0, 5))}},
                "ai_signal": {"ai_composite_signal": {"score": float(np.clip(change_rate * 500 + rng.normal(0, 15), -50, 50))}},
                "coinmarketcap": {"btc_dominance": float(np.clip(55 + rng.normal(0, 3), 40, 70))}},
            "external_signal": {"total_score": float(np.clip(change_rate * 400 + rng.normal(0, 10), -50, 50))},
        }

        # 포트폴리오
        total_value = self._portfolio_value(price)
        btc_eval = self.btc_balance * price
        pnl_pct = ((price - self.avg_buy_price) / self.avg_buy_price * 100) if self.avg_buy_price > 0 else 0
        portfolio = {
            "krw_balance": self.krw_balance,
            "holdings": [{
                "currency": "BTC",
                "balance": self.btc_balance,
                "avg_buy_price": self.avg_buy_price,
                "eval_amount": btc_eval,
                "profit_loss_pct": pnl_pct,
            }] if self.btc_balance > 0 else [],
            "total_eval": total_value,
        }

        # 에이전트 상태 (동적)
        # 위험 점수: MDD 기반, 기회 점수: RSI 기반
        values = self.total_value_history
        recent_dd = 0
        if len(values) > 10:
            peak = max(values[-10:])
            recent_dd = (peak - values[-1]) / peak * 100
        agent_state = {
            "danger_score": min(80, 20 + recent_dd * 5),
            "opportunity_score": max(10, 70 - abs(rsi - 50)),
            "cascade_risk": min(60, 10 + recent_dd * 3),
            "consecutive_losses": 0,
            "hours_since_last_trade": min(72, max(1, self.reward_calc.steps_since_trade)),
            "daily_trade_count": self.trade_count,
        }

        return self.encoder.encode(market_data, external_data, portfolio, agent_state)

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
            print(self.render.__func__(self))

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

        # 관측 공간 확장
        total_dim = OBSERVATION_DIM + llm_embedding_dim
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(total_dim,), dtype=np.float32
        )

    def set_llm_embedding(self, embedding: np.ndarray):
        """외부에서 LLM 임베딩 주입"""
        self.llm_embedding = embedding.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        base_obs = super()._get_observation()
        return np.concatenate([base_obs, self.llm_embedding])

"""BitcoinTradingEnvV2 — 6가지 훈련 갭을 해결하는 강화 환경

개선사항:
  1. 정책 붕괴 방지: 행동 다양성 보상 + 거래 보너스 강화
  2. 폭락/블랙스완: 합성 급락 시나리오 주입
  3. 외부 데이터: 가격 패턴 기반 현실적 시뮬레이션 (노이즈→상관 모델)
  4. 시장 국면: 캔들 데이터에 regime 라벨 부여
  5. DCA/손절: 보상에 단계적 진입/퇴출 인센티브
  6. 시간 패턴: 주말/야간 거래량 감소 반영
"""

import copy
import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM
from rl_hybrid.rl.reward import RewardCalculator, TRANSACTION_COST
from rl_hybrid.rl.data_loader import HistoricalDataLoader

logger = logging.getLogger("rl.environment_v2")


# ── 시장 국면 분류 ──────────────────────────────────

def classify_regime(candles: list[dict], idx: int, lookback: int = 20) -> str:
    """최근 lookback 봉의 가격 추세로 국면을 분류한다."""
    start = max(0, idx - lookback)
    prices = [c["close"] for c in candles[start:idx + 1]]
    if len(prices) < 5:
        return "sideways"

    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    total_return = (prices[-1] - prices[0]) / prices[0]
    volatility = np.std(returns) if returns else 0.01

    if total_return > 0.03 and np.mean(returns) > 0:
        return "bull"
    elif total_return < -0.03 and np.mean(returns) < 0:
        return "bear"
    elif volatility > 0.015:
        return "volatile"
    else:
        return "sideways"


# ── 폭락 시나리오 주입 ──────────────────────────────

def inject_crashes(candles: list[dict], rng: np.random.Generator,
                   crash_prob: float = 0.03, min_drop: float = -0.15,
                   max_drop: float = -0.05) -> list[dict]:
    """캔들 데이터에 합성 폭락 시나리오를 삽입한다.

    crash_prob: 각 캔들에서 폭락이 시작될 확률
    min_drop/max_drop: 폭락 폭 범위 (음수)
    """
    candles = copy.deepcopy(candles)
    i = 0
    while i < len(candles) - 10:
        if rng.random() < crash_prob:
            # 폭락 크기 및 지속 기간
            drop_pct = rng.uniform(min_drop, max_drop)
            duration = rng.integers(3, 12)  # 3~12봉에 걸쳐 하락
            recovery_duration = rng.integers(5, 20)  # 회복 기간

            base_price = candles[i]["close"]

            # 급락 구간
            for j in range(duration):
                if i + j >= len(candles):
                    break
                progress = (j + 1) / duration
                # 급락은 초반에 급격, 후반에 완만 (제곱근 커브)
                drop = drop_pct * np.sqrt(progress)
                c = candles[i + j]
                factor = 1 + drop
                c["close"] = base_price * factor
                c["low"] = min(c["low"], c["close"] * 0.995)
                c["high"] = max(c["open"], c["close"])
                c["volume"] = c["volume"] * (1 + abs(drop) * 10)  # 폭락 시 거래량 급증
                # change_rate 재계산
                if i + j > 0:
                    prev_close = candles[i + j - 1]["close"]
                    c["change_rate"] = (c["close"] - prev_close) / prev_close

            # 부분 회복 구간 (50~80% 회복)
            bottom_price = candles[min(i + duration - 1, len(candles) - 1)]["close"]
            recovery_target = base_price * (1 + drop_pct * rng.uniform(0.2, 0.5))

            for j in range(recovery_duration):
                idx = i + duration + j
                if idx >= len(candles):
                    break
                progress = (j + 1) / recovery_duration
                c = candles[idx]
                c["close"] = bottom_price + (recovery_target - bottom_price) * progress
                c["high"] = c["close"] * (1 + rng.uniform(0, 0.01))
                c["low"] = c["close"] * (1 - rng.uniform(0, 0.005))
                if idx > 0:
                    prev_close = candles[idx - 1]["close"]
                    c["change_rate"] = (c["close"] - prev_close) / prev_close

            i += duration + recovery_duration + rng.integers(20, 50)
        else:
            i += 1

    return candles


# ── 현실적 외부 데이터 시뮬레이션 ──────────────────

def simulate_realistic_external(candle: dict, candles: list[dict],
                                 idx: int, rng: np.random.Generator) -> dict:
    """가격 패턴과 지표에 기반한 현실적 외부 데이터를 생성한다.

    기존: 단순 노이즈
    개선: 가격 추세/변동성/RSI와 상관관계 있는 시뮬레이션
    """
    price = candle["close"]
    rsi = candle.get("rsi_14", 50)
    change_rate = candle.get("change_rate", 0)

    # 최근 24봉 수익률
    start = max(0, idx - 24)
    prices_24h = [c["close"] for c in candles[start:idx + 1]]
    ret_24h = (prices_24h[-1] - prices_24h[0]) / prices_24h[0] if len(prices_24h) > 1 else 0

    # 최근 변동성 (ATR proxy)
    recent_changes = [abs(c.get("change_rate", 0)) for c in candles[max(0, idx - 14):idx + 1]]
    volatility = np.mean(recent_changes) if recent_changes else 0.01

    # ── FGI: RSI + 24h 수익률 + 변동성 기반 ──
    # 실제 FGI 패턴: 급락 시 극공포, 상승 시 탐욕, 지속 상승 시 극탐욕
    fgi_base = 50 + ret_24h * 500 + (rsi - 50) * 0.5
    # 변동성 높으면 FGI 하락 (공포 증가)
    fgi_base -= volatility * 300
    # 관성 (이전 FGI에서 크게 변하지 않음)
    fgi = np.clip(fgi_base + rng.normal(0, 3), 5, 95)

    # ── 뉴스 감성: 가격 추세 + 지연 효과 ──
    # 뉴스는 가격보다 약간 늦게 반응
    lagged_ret = 0
    if idx >= 6:
        lagged_prices = [c["close"] for c in candles[idx - 6:idx + 1]]
        lagged_ret = (lagged_prices[-1] - lagged_prices[0]) / lagged_prices[0]
    news_sentiment = np.clip(lagged_ret * 400 + rng.normal(0, 5), -50, 50)

    # ── 고래 점수: 급등/급락 시 활성화 ──
    whale_activity = abs(change_rate) * 300 + rng.normal(0, 2)
    whale_score = np.clip(whale_activity, -10, 10)

    # ── 펀딩 레이트: RSI/포지션 과밀도 반영 ──
    # RSI 높으면 롱 과밀 → 양수 펀딩, RSI 낮으면 숏 과밀 → 음수 펀딩
    funding = (rsi - 50) * 0.0015 + rng.normal(0, 0.003)
    funding = np.clip(funding, -0.03, 0.03)

    # ── 롱숏비율: RSI 기반 ──
    ls_ratio = 1.0 + (rsi - 50) * 0.008 + rng.normal(0, 0.1)
    ls_ratio = np.clip(ls_ratio, 0.5, 2.0)

    # ── 김치 프리미엄: 가격 추세 + 변동성 ──
    # 상승장에서 양수, 하락장에서 음수/축소
    kp = ret_24h * 80 + rng.normal(0.5, 0.8)
    kp = np.clip(kp, -3, 8)

    # ── 매크로: 느린 추세 반영 ──
    if idx >= 48:
        weekly_prices = [c["close"] for c in candles[idx - 48:idx + 1]]
        weekly_ret = (weekly_prices[-1] - weekly_prices[0]) / weekly_prices[0]
    else:
        weekly_ret = ret_24h
    macro = np.clip(weekly_ret * 200 + rng.normal(0, 3), -20, 20)

    # ── AI 복합 시그널: 모든 지표 종합 ──
    ai_signal = (fgi - 50) * 0.3 + news_sentiment * 0.2 + macro * 0.3 + whale_score * 0.2
    ai_signal = np.clip(ai_signal + rng.normal(0, 3), -50, 50)

    # ── BTC 도미넌스: 느린 변화 ──
    btc_dom = 55 + weekly_ret * 50 + rng.normal(0, 1)
    btc_dom = np.clip(btc_dom, 40, 70)

    # ── 시간 패턴 반영 ──
    timestamp = candle.get("timestamp", 0)
    if timestamp:
        from datetime import datetime, timezone
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            else:
                ts_num = float(timestamp)
                dt = datetime.fromtimestamp(ts_num / 1000 if ts_num > 1e12 else ts_num, tz=timezone.utc)
        except (ValueError, OSError, TypeError):
            dt = None
        if dt is None:
            return external_data
        hour = dt.hour
        weekday = dt.weekday()
        is_weekend = weekday >= 5

        # 주말: 거래량 감소 → FGI 안정, 김프 축소
        if is_weekend:
            fgi = fgi * 0.9 + 50 * 0.1  # 50쪽으로 수렴
            kp *= 0.7
            whale_score *= 0.5  # 고래 활동 감소

        # 야간(UTC 14-22 = KST 23-07): 거래량 감소
        if 14 <= hour <= 22:
            whale_score *= 0.6

    # ── Fusion 점수 ──
    fusion_score = (fgi - 50) * 0.2 + news_sentiment * 0.15 + whale_score * 0.15 + macro * 0.25 + (50 - rsi) * 0.1
    fusion_score = np.clip(fusion_score, -50, 50)

    return {
        "sources": {
            "fear_greed": {"current": {"value": float(fgi)}},
            "news_sentiment": {"sentiment_score": float(news_sentiment)},
            "whale_tracker": {"whale_score": {"score": float(whale_score)}},
            "binance_sentiment": {
                "funding_rate": {"current_rate": float(funding)},
                "top_trader_long_short": {"current_ratio": float(ls_ratio)},
                "kimchi_premium": {"premium_pct": float(kp)},
            },
            "macro": {"analysis": {"macro_score": float(macro)}},
            "ai_signal": {"ai_composite_signal": {"score": float(ai_signal)}},
            "coinmarketcap": {"btc_dominance": float(btc_dom)},
            "eth_btc": {"eth_btc_z_score": float(rng.normal(0, 0.5))},
        },
        "external_signal": {"total_score": float(fusion_score)},
    }


# ── 강화 환경 ──────────────────────────────────────

class BitcoinTradingEnvV2(gym.Env):
    """6가지 훈련 갭을 해결하는 강화된 비트코인 트레이딩 환경.

    v1 대비 개선:
      - 합성 폭락 시나리오 주입
      - 현실적 외부 데이터 시뮬레이션
      - 행동 다양성 보상 (정책 붕괴 방지)
      - 시장 국면 라벨링
      - DCA 단계적 진입 보상
      - 주말/야간 패턴 반영
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        candles: list[dict] = None,
        initial_balance: float = 10_000_000,
        max_steps: int = None,
        lookback: int = 24,
        render_mode: str = None,
        # V2 옵션
        crash_injection: bool = True,
        crash_prob: float = 0.02,
        regime_aware: bool = True,
        dca_reward: bool = True,
        anti_collapse: bool = True,
    ):
        super().__init__()

        if candles is None:
            loader = HistoricalDataLoader()
            raw = loader.load_candles(days=180, interval="1h")
            candles = loader.compute_indicators(raw)

        self.original_candles = candles
        self.initial_balance = initial_balance
        self.lookback = lookback
        self.render_mode = render_mode
        self.max_steps_cfg = max_steps

        # V2 옵션
        self.crash_injection = crash_injection
        self.crash_prob = crash_prob
        self.regime_aware = regime_aware
        self.dca_reward = dca_reward
        self.anti_collapse = anti_collapse

        # 공간 정의 (42차원 유지 — 기존 모델 호환)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.encoder = StateEncoder()
        self.reward_calc = RewardCalculator()

        # 상태 변수
        self.candles = candles
        self.current_step = 0
        self.start_idx = lookback
        self.end_idx = len(candles) - 1
        self.krw_balance = 0.0
        self.btc_balance = 0.0
        self.avg_buy_price = 0.0
        self.prev_action = 0.0
        self.total_value_history = []
        self.action_history = []
        self.trade_count = 0
        self.regime = "sideways"
        self.steps_no_trade = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 폭락 주입 (에피소드마다 다른 폭락 패턴)
        if self.crash_injection:
            self.candles = inject_crashes(
                self.original_candles, self.np_random,
                crash_prob=self.crash_prob,
            )
        else:
            self.candles = copy.deepcopy(self.original_candles)

        self.start_idx = self.lookback
        self.end_idx = len(self.candles) - 1
        if self.max_steps_cfg:
            self.end_idx = min(self.start_idx + self.max_steps_cfg, self.end_idx)

        # 랜덤 시작점
        max_start = max(self.start_idx, self.end_idx - 500)
        self.current_step = self.np_random.integers(self.start_idx, max_start + 1)

        self.krw_balance = self.initial_balance
        self.btc_balance = 0.0
        self.avg_buy_price = 0.0
        self.prev_action = 0.0
        self.total_value_history = []
        self.action_history = []
        self.trade_count = 0
        self.steps_no_trade = 0

        self.reward_calc = RewardCalculator()

        obs = self._get_observation()
        self.regime = classify_regime(self.candles, self.current_step)

        return obs, {"regime": self.regime}

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        price = self.candles[self.current_step]["close"]
        prev_value = self._portfolio_value(price)

        # 행동 실행 (v1과 동일)
        target_btc_ratio = (action_val + 1) / 2
        current_btc_value = self.btc_balance * price
        total_value = self._portfolio_value(price)
        target_btc_value = total_value * target_btc_ratio
        diff = target_btc_value - current_btc_value

        traded = False
        trade_amount = 0.0

        if abs(diff) > total_value * 0.01:  # 1% 이상 변동만
            if diff > 0:  # 매수
                cost = diff * (1 + TRANSACTION_COST)
                if cost <= self.krw_balance:
                    btc_bought = diff / price
                    # 평균매수가 업데이트
                    if self.btc_balance > 0:
                        total_cost = self.avg_buy_price * self.btc_balance + price * btc_bought
                        self.btc_balance += btc_bought
                        self.avg_buy_price = total_cost / self.btc_balance
                    else:
                        self.btc_balance = btc_bought
                        self.avg_buy_price = price
                    self.krw_balance -= cost
                    traded = True
                    trade_amount = diff
            else:  # 매도
                btc_to_sell = abs(diff) / price
                btc_to_sell = min(btc_to_sell, self.btc_balance)
                if btc_to_sell > 0:
                    proceeds = btc_to_sell * price * (1 - TRANSACTION_COST)
                    self.btc_balance -= btc_to_sell
                    self.krw_balance += proceeds
                    traded = True
                    trade_amount = abs(diff)
                    if self.btc_balance < 1e-10:
                        self.btc_balance = 0
                        self.avg_buy_price = 0

        if traded:
            self.trade_count += 1
            self.steps_no_trade = 0
        else:
            self.steps_no_trade += 1

        self.action_history.append(action_val)
        self.prev_action = action_val

        # 다음 스텝
        self.current_step += 1
        new_price = self.candles[self.current_step]["close"]
        new_value = self._portfolio_value(new_price)
        self.total_value_history.append(new_value)

        # ── 보상 계산 (V2 강화) ──
        reward_info = self.reward_calc.calculate(
            prev_portfolio_value=prev_value,
            curr_portfolio_value=new_value,
            action=action_val,
            prev_action=self.prev_action,
            step=self.current_step - self.start_idx,
        )
        reward = reward_info["reward"] if isinstance(reward_info, dict) else float(reward_info)

        # [개선 1] 정책 붕괴 방지 — 행동 다양성 보상
        if self.anti_collapse:
            # 너무 오래 거래 안 하면 페널티 강화
            if self.steps_no_trade > 8:
                reward -= 0.02 * (self.steps_no_trade - 8)

            # 거래 실행 시 보너스 (수익성과 무관)
            if traded:
                reward += 0.08

            # 행동 변화 보너스 (같은 행동 반복 시 감소)
            if len(self.action_history) >= 3:
                recent = self.action_history[-3:]
                action_std = np.std(recent)
                if action_std > 0.1:
                    reward += 0.03  # 다양한 행동

        # [개선 2] 폭락 대응 보상
        price_change = (new_price - price) / price
        if price_change < -0.03:  # 3% 이상 급락
            # 현금 비중이 높으면 보상 (위기 회피)
            cash_ratio = self.krw_balance / new_value if new_value > 0 else 0
            if cash_ratio > 0.5:
                reward += 0.15  # 폭락에서 현금 보유 = 좋은 판단
            # BTC 과다 보유 상태에서 매도했으면 보상
            if traded and action_val < -0.3:
                reward += 0.1

        # [개선 5] DCA 단계적 진입 보상
        if self.dca_reward and traded:
            btc_ratio = (self.btc_balance * new_price) / new_value if new_value > 0 else 0
            # 한 번에 올인하지 않고 단계적 진입 보상
            if 0.2 < btc_ratio < 0.8:
                reward += 0.03  # 적절한 포지션 크기

        # [개선 6] 주말 패턴
        timestamp = self.candles[self.current_step].get("timestamp", 0)
        if timestamp:
            from datetime import datetime, timezone
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    ts_num = float(timestamp)
                    dt = datetime.fromtimestamp(
                        ts_num / 1000 if ts_num > 1e12 else ts_num,
                        tz=timezone.utc
                    )
                if dt.weekday() >= 5 and traded:
                    # 주말 큰 거래 페널티 (유동성 낮음)
                    if trade_amount > self.initial_balance * 0.1:
                        reward -= 0.02
            except (ValueError, OSError):
                pass

        # 국면 업데이트
        if self.regime_aware and self.current_step % 10 == 0:
            self.regime = classify_regime(self.candles, self.current_step)

        # 종료 조건
        terminated = self.current_step >= self.end_idx - 1
        truncated = False

        # 파산 체크
        if new_value < self.initial_balance * 0.1:
            reward = -1.0
            terminated = True

        obs = self._get_observation()
        info = {
            "total_value": new_value,
            "trade_count": self.trade_count,
            "regime": self.regime,
            "profit_pct": (new_value - self.initial_balance) / self.initial_balance * 100,
        }

        return obs, float(reward), terminated, truncated, info

    def _portfolio_value(self, price: float) -> float:
        return self.krw_balance + self.btc_balance * price

    def _get_observation(self) -> np.ndarray:
        candle = self.candles[self.current_step]
        price = candle["close"]

        market_data = {
            "ticker": {
                "trade_price": price,
                "signed_change_rate": candle.get("change_rate", 0),
                "acc_trade_volume_24h": candle.get("volume", 0),
            },
            "indicators": {
                "rsi_14": candle.get("rsi_14", 50),
                "sma_20": candle.get("sma_20", price),
                "sma_50": candle.get("sma_50", price),
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

        # [개선 3] 현실적 외부 데이터 시뮬레이션
        external_data = simulate_realistic_external(
            candle, self.candles, self.current_step, self.np_random
        )

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

        # 에이전트 상태
        values = self.total_value_history
        recent_dd = 0
        if len(values) > 10:
            peak = max(values[-10:])
            recent_dd = (peak - values[-1]) / peak * 100

        agent_state = {
            "danger_score": min(80, 20 + recent_dd * 5),
            "opportunity_score": max(10, 70 - abs(candle.get("rsi_14", 50) - 50)),
            "cascade_risk": min(60, 10 + recent_dd * 3),
            "consecutive_losses": 0,
            "hours_since_last_trade": min(72, max(1, self.steps_no_trade)),
            "daily_trade_count": self.trade_count,
        }

        return self.encoder.encode(market_data, external_data, portfolio, agent_state)

    def _get_info(self) -> dict:
        price = self.candles[self.current_step]["close"]
        total = self._portfolio_value(price)
        return {
            "total_value": total,
            "trade_count": self.trade_count,
            "regime": self.regime,
            "profit_pct": (total - self.initial_balance) / self.initial_balance * 100,
        }

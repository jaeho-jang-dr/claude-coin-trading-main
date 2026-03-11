"""상태 인코더 — 시장 데이터 + 외부 지표 + 포트폴리오 → RL 관측 벡터

원시 데이터를 고정 길이 정규화 벡터로 변환한다.
모든 값은 대략 [-1, 1] 또는 [0, 1] 범위로 정규화한다.
"""

import numpy as np
from typing import Optional

# 관측 벡터 구성 (총 42차원)
FEATURE_SPEC = {
    # === 가격 특성 (7) ===
    "price_change_1h": (-0.10, 0.10),      # %
    "price_change_4h": (-0.15, 0.15),
    "price_change_24h": (-0.20, 0.20),
    "price_vs_sma20": (-0.15, 0.15),       # (price - sma20) / sma20
    "price_vs_sma50": (-0.20, 0.20),
    "price_vs_boll_upper": (-0.10, 0.10),  # (price - upper) / upper
    "price_vs_boll_lower": (-0.10, 0.10),

    # === 기술 지표 (10) ===
    "rsi_14": (0, 100),
    "rsi_4h": (0, 100),
    "macd_histogram": (-500, 500),
    "stochastic_k": (0, 100),
    "stochastic_d": (0, 100),
    "adx": (0, 100),
    "adx_plus_di": (0, 50),
    "adx_minus_di": (0, 50),
    "atr_pct": (0, 5),                     # ATR / price * 100
    "bollinger_width": (0, 0.15),          # (upper - lower) / middle

    # === 거래량 (4) ===
    "volume_change_ratio": (0, 5),          # recent / avg
    "orderbook_ratio": (0, 3),              # bid / ask
    "trade_pressure_ratio": (0, 3),         # buy_vol / sell_vol
    "whale_buy_ratio": (0, 1),

    # === 감성/외부 지표 (10) ===
    "fgi": (0, 100),
    "news_sentiment": (-100, 100),
    "whale_score": (-30, 30),
    "funding_rate": (-0.1, 0.1),
    "long_short_ratio": (0.5, 3.0),
    "kimchi_premium": (-5, 10),
    "macro_score": (-30, 30),
    "eth_btc_z_score": (-3, 3),
    "ai_composite_signal": (-100, 100),
    "btc_dominance": (30, 70),

    # === 에이전트 상태 (5) ===
    "danger_score": (0, 100),
    "opportunity_score": (0, 100),
    "cascade_risk": (0, 100),
    "fusion_score": (-100, 100),
    "consecutive_losses": (0, 10),

    # === 포트폴리오 (6) ===
    "position_ratio": (0, 1),               # BTC 평가액 / 총 자산
    "unrealized_pnl": (-0.30, 0.30),        # 미실현 수익률
    "cash_ratio": (0, 1),                   # KRW / 총 자산
    "avg_cost_vs_price": (-0.20, 0.20),     # (현재가 - 평단) / 평단
    "hours_since_last_trade": (0, 72),
    "daily_trade_count": (0, 10),
}

OBSERVATION_DIM = len(FEATURE_SPEC)
FEATURE_NAMES = list(FEATURE_SPEC.keys())


class StateEncoder:
    """원시 데이터 → 정규화 관측 벡터 변환기"""

    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.obs_dim = OBSERVATION_DIM
        self._mins = np.array([v[0] for v in FEATURE_SPEC.values()], dtype=np.float32)
        self._maxs = np.array([v[1] for v in FEATURE_SPEC.values()], dtype=np.float32)
        self._ranges = self._maxs - self._mins
        self._ranges[self._ranges == 0] = 1  # 0 나누기 방지

    def encode(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        agent_state: dict = None,
    ) -> np.ndarray:
        """원시 데이터를 정규화 벡터로 변환

        Returns:
            np.ndarray shape (42,), 값 범위 [0, 1]
        """
        raw = self._extract_raw_features(
            market_data, external_data, portfolio, agent_state
        )
        return self._normalize(raw)

    def _extract_raw_features(
        self,
        md: dict,       # market_data
        ext: dict,      # external_data
        pf: dict,       # portfolio
        ag: dict = None, # agent_state
    ) -> np.ndarray:
        """원시 특성 추출"""
        ag = ag or {}
        indicators = md.get("indicators", {})
        indicators_4h = md.get("indicators_4h", {})
        orderbook = md.get("orderbook", {})
        trade_pressure = md.get("trade_pressure", {})
        macd = indicators.get("macd", {})
        bollinger = indicators.get("bollinger", {})
        stoch = indicators.get("stochastic", {})
        adx = indicators.get("adx", {})
        eth_btc = md.get("eth_btc_analysis", {})

        # 외부 데이터 소스 추출
        sources = ext.get("sources", ext)
        fgi_data = sources.get("fear_greed", {})
        news_data = sources.get("news_sentiment", {})
        whale_data = sources.get("whale_tracker", {})
        binance = sources.get("binance_sentiment", {})
        macro = sources.get("macro", {})
        ai_sig = sources.get("ai_signal", {})
        cmc = sources.get("coinmarketcap", {})
        ext_signal = ext.get("external_signal", {})

        # 가격 정보
        price = float(md.get("current_price", 0))
        sma20 = float(indicators.get("sma_20", price) or price)
        sma50 = float(indicators.get("sma_50", price) or price)
        boll_upper = float(bollinger.get("upper", price) or price)
        boll_lower = float(bollinger.get("lower", price) or price)
        boll_middle = float(bollinger.get("middle", price) or price)

        # 포트폴리오 추출
        holdings = pf.get("holdings", [])
        btc_holding = next((h for h in holdings if h.get("currency") == "BTC"), {})
        total_eval = float(pf.get("total_eval", 1))
        krw = float(pf.get("krw_balance", 0))
        btc_eval = float(btc_holding.get("eval_amount", 0))
        avg_buy = float(btc_holding.get("avg_buy_price", price) or price)

        # 안전한 나눗셈 헬퍼
        def safe_div(a, b, default=0.0):
            return float(a) / float(b) if b else default

        raw = np.array([
            # 가격 특성
            float(md.get("change_rate_24h", 0)) * 0.3,  # 1h 근사 (24h의 1/8)
            float(md.get("change_rate_24h", 0)) * 0.5,  # 4h 근사
            float(md.get("change_rate_24h", 0)),
            safe_div(price - sma20, sma20),
            safe_div(price - sma50, sma50),
            safe_div(price - boll_upper, boll_upper),
            safe_div(price - boll_lower, boll_lower),

            # 기술 지표
            float(indicators.get("rsi_14", 50) or 50),
            float(indicators_4h.get("rsi_14", 50) or 50),
            float(macd.get("histogram", 0) or 0),
            float(stoch.get("k", 50) or 50),
            float(stoch.get("d", 50) or 50),
            float(adx.get("adx", 25) or 25),
            float(adx.get("plus_di", 20) or 20),
            float(adx.get("minus_di", 20) or 20),
            safe_div(float(indicators.get("atr", 0) or 0), price) * 100,
            safe_div(boll_upper - boll_lower, boll_middle),

            # 거래량
            1.0,  # volume_change_ratio (default, 히스토리 필요)
            float(orderbook.get("ratio", 1.0) or 1.0),
            safe_div(
                float(trade_pressure.get("buy_volume", 1)),
                float(trade_pressure.get("sell_volume", 1)),
                1.0,
            ),
            float(whale_data.get("whale_score", {}).get("score", 0) + 30) / 60,

            # 감성/외부 지표
            float(fgi_data.get("current", {}).get("value", 50) if isinstance(fgi_data.get("current"), dict) else fgi_data.get("value", 50)),
            float(news_data.get("sentiment_score", 0) if isinstance(news_data, dict) else 0),
            float(whale_data.get("whale_score", {}).get("score", 0) if isinstance(whale_data, dict) else 0),
            float(binance.get("funding_rate", {}).get("current_rate", 0) if isinstance(binance.get("funding_rate"), dict) else 0),
            float(binance.get("top_trader_long_short", {}).get("current_ratio", 1.0) if isinstance(binance.get("top_trader_long_short"), dict) else 1.0),
            float(binance.get("kimchi_premium", {}).get("premium_pct", 0) if isinstance(binance.get("kimchi_premium"), dict) else 0),
            float(macro.get("analysis", {}).get("macro_score", 0) if isinstance(macro.get("analysis"), dict) else 0),
            float(eth_btc.get("eth_btc_z_score", 0) or 0),
            float(ai_sig.get("ai_composite_signal", {}).get("score", 0) if isinstance(ai_sig.get("ai_composite_signal"), dict) else 0),
            float(cmc.get("btc_dominance", 50) or 50),

            # 에이전트 상태
            float(ag.get("danger_score", 30)),
            float(ag.get("opportunity_score", 30)),
            float(ag.get("cascade_risk", 20)),
            float(ext_signal.get("total_score", 0)),
            float(ag.get("consecutive_losses", 0)),

            # 포트폴리오
            safe_div(btc_eval, total_eval) if total_eval > 0 else 0,
            float(btc_holding.get("profit_loss_pct", 0) or 0) / 100,
            safe_div(krw, total_eval) if total_eval > 0 else 1,
            safe_div(price - avg_buy, avg_buy) if avg_buy > 0 else 0,
            float(ag.get("hours_since_last_trade", 24)),
            float(ag.get("daily_trade_count", 0)),
        ], dtype=np.float32)

        return raw

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """Min-Max 정규화 → [0, 1] 범위"""
        normalized = (raw - self._mins) / self._ranges
        return np.clip(normalized, 0, 1)

    def decode_feature(self, obs: np.ndarray, feature_name: str) -> float:
        """정규화된 관측값을 원래 스케일로 복원"""
        idx = self.feature_names.index(feature_name)
        return obs[idx] * self._ranges[idx] + self._mins[idx]

"""
🔥 공격적 전략 에이전트

빈번한 매매, 빠른 익절/손절. 고위험 고수익.
매수 임계: 40점, 손절: -3%, 강제 손절: -7%
v2 조정(2026-03-11): 45→40
"""

from __future__ import annotations

from agents.base_agent import BaseStrategyAgent, Decision


class AggressiveAgent(BaseStrategyAgent):

    name = "aggressive"
    emoji = "🔥"
    description = "공격적 — 빈번한 매매, 빠른 익절/손절"

    # 매수 조건 (공격적: 뉴스 무관 자동 20점)
    fgi_threshold = 60
    rsi_threshold = 50
    sma_deviation_pct = -1.0
    buy_score_threshold = 40
    macd_bonus = False

    # 매도 조건
    target_profit_pct = 7.0
    stop_loss_pct = -3.0
    forced_stop_loss_pct = -7.0
    sell_fgi_threshold = 65
    sell_rsi_threshold = 60

    # 매매 규모
    max_trade_ratio = 0.20
    max_daily_trades = 7
    weekend_reduction = 0.0  # 주말 축소 없음

    # 공격적은 뉴스 무관
    news_points = 20  # 자동 부여

    def decide(self, market_data: dict, external_signal: dict, portfolio: dict,
               drop_context: dict | None = None) -> Decision:
        ind = self._extract_indicators(market_data)
        external_bonus = external_signal.get("strategy_bonus", 0)
        ai_score = market_data.get("ai_composite_signal", {}).get("score", 0)
        fgi = market_data.get("fear_greed", {}).get("value", 50)

        # 공격적 전략: 뉴스 무관 (news_negative=False 고정)
        buy_score = self.calculate_buy_score(
            fgi=fgi,
            rsi=ind["rsi"],
            sma_deviation=ind["sma_deviation"],
            news_negative=False,  # 항상 20점 자동 부여
            external_bonus=external_bonus,
        )

        # 보유 중이면 매도 조건 먼저
        btc_holding = portfolio.get("btc", {})
        if btc_holding.get("balance", 0) > 0:
            profit_pct = btc_holding.get("profit_pct", 0)
            total_eval = portfolio.get("total_eval", 0)
            btc_eval = btc_holding.get("eval_amount", 0)
            btc_position_ratio = btc_eval / total_eval if total_eval > 0 else 0

            sell_eval = self.evaluate_sell(
                profit_pct=profit_pct,
                current_fgi=fgi,
                current_rsi=ind["rsi"],
                buy_score=buy_score,
                ai_signal_score=ai_score,
                drop_context=drop_context,
                btc_position_ratio=btc_position_ratio,
            )
            if sell_eval:
                action = sell_eval["action"]
                if action == "sell":
                    return Decision(
                        decision="sell",
                        confidence=0.8,
                        reason=sell_eval["reason"],
                        buy_score=buy_score,
                        trade_params={
                            "side": "ask",
                            "market": "KRW-BTC",
                            "volume": btc_holding.get("balance", 0),
                        },
                        external_signal=external_signal,
                        agent_name=f"{self.emoji} {self.name}",
                    )
                elif action == "sell_partial":
                    sell_volume = round(btc_holding.get("balance", 0) * sell_eval.get("sell_ratio", 1/3), 8)
                    return Decision(
                        decision="sell",
                        confidence=0.75,
                        reason=sell_eval["reason"],
                        buy_score=buy_score,
                        trade_params={
                            "side": "ask",
                            "market": "KRW-BTC",
                            "volume": sell_volume,
                            "is_partial": True,
                        },
                        external_signal=external_signal,
                        agent_name=f"{self.emoji} {self.name}",
                    )
                elif action == "dca":
                    total_krw = portfolio.get("krw_balance", 0)
                    dca_amount = min(
                        int(btc_holding.get("avg_buy_price", 0) * btc_holding.get("balance", 0) * self.dca_max_ratio),
                        self._calculate_trade_amount(total_krw, external_bonus),
                    )
                    return Decision(
                        decision="buy",
                        confidence=0.6,
                        reason=sell_eval["reason"],
                        buy_score=buy_score,
                        trade_params={"side": "bid", "market": "KRW-BTC", "amount": dca_amount, "is_dca": True},
                        external_signal=external_signal,
                        agent_name=f"{self.emoji} {self.name}",
                    )

        # 매수 판단 (공격적: AI 필터 완화)
        if buy_score["result"] == "buy":
            total_krw = portfolio.get("krw_balance", 0)
            amount = self._calculate_trade_amount(total_krw, external_bonus)

            return Decision(
                decision="buy",
                confidence=min(0.9, buy_score["total"] / 100),
                reason=f"매수 점수 {buy_score['total']}점 >= {self.buy_score_threshold}점 충족",
                buy_score=buy_score,
                trade_params={"side": "bid", "market": "KRW-BTC", "amount": amount},
                external_signal=external_signal,
                agent_name=f"{self.emoji} {self.name}",
            )

        return Decision(
            decision="hold",
            confidence=0.7,
            reason=f"매수 점수 {buy_score['total']}점 < {self.buy_score_threshold}점. 관망.",
            buy_score=buy_score,
            trade_params={},
            external_signal=external_signal,
            agent_name=f"{self.emoji} {self.name}",
        )

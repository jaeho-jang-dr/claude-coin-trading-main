"""
⚖️ 보통 전략 에이전트

조정장에서도 매수 가능. 균형잡힌 수익/리스크.
매수 임계: 55점, 손절: -5%, 강제 손절: -10%
"""

from __future__ import annotations

from agents.base_agent import BaseStrategyAgent, Decision


class ModerateAgent(BaseStrategyAgent):

    name = "moderate"
    emoji = "⚖️"
    description = "보통 — 조정장 매수, 균형 수익/리스크"

    # 매수 조건
    fgi_threshold = 45
    rsi_threshold = 40
    sma_deviation_pct = -3.0
    buy_score_threshold = 55
    macd_bonus = True

    # 매도 조건
    target_profit_pct = 10.0
    stop_loss_pct = -5.0
    forced_stop_loss_pct = -10.0
    sell_fgi_threshold = 70
    sell_rsi_threshold = 65

    # 매매 규모
    max_trade_ratio = 0.15
    max_daily_trades = 5
    weekend_reduction = 0.30

    def decide(self, market_data: dict, external_signal: dict, portfolio: dict,
               drop_context: dict | None = None) -> Decision:
        ind = self._extract_indicators(market_data)
        external_bonus = external_signal.get("strategy_bonus", 0)
        ai_score = market_data.get("ai_composite_signal", {}).get("score", 0)
        fgi = market_data.get("fear_greed", {}).get("value", 50)

        news = market_data.get("news", {})
        news_negative = news.get("overall_sentiment", "neutral") == "negative"

        macd = ind.get("macd", {})
        macd_golden = macd.get("histogram", 0) > 0 and macd.get("signal_cross", False)

        buy_score = self.calculate_buy_score(
            fgi=fgi,
            rsi=ind["rsi"],
            sma_deviation=ind["sma_deviation"],
            news_negative=news_negative,
            external_bonus=external_bonus,
            macd_golden_cross=macd_golden,
        )

        # 보유 중이면 매도 조건 먼저
        btc_holding = portfolio.get("btc", {})
        if btc_holding.get("balance", 0) > 0:
            profit_pct = btc_holding.get("profit_pct", 0)
            sell_eval = self.evaluate_sell(
                profit_pct=profit_pct,
                current_fgi=fgi,
                current_rsi=ind["rsi"],
                buy_score=buy_score,
                ai_signal_score=ai_score,
                drop_context=drop_context,
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

        # 매수 판단
        if buy_score["result"] == "buy":
            if ai_score < 0 and fgi > 20:
                return Decision(
                    decision="hold",
                    confidence=0.5,
                    reason=f"매수 점수 {buy_score['total']}점 충족이나 AI 시그널 음수({ai_score}) → 보류",
                    buy_score=buy_score,
                    trade_params={},
                    external_signal=external_signal,
                    agent_name=f"{self.emoji} {self.name}",
                )

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

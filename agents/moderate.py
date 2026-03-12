"""
⚖️ 보통 전략 에이전트

조정장에서도 매수 가능. 균형잡힌 수익/리스크.
매수 임계: 50점, 손절: -5%, 강제 손절: -10%
v2 조정(2026-03-11): 55→50, SMA -3→-2
"""

from __future__ import annotations

from agents.base_agent import BaseStrategyAgent, Decision


class ModerateAgent(BaseStrategyAgent):

    name = "moderate"
    emoji = "⚖️"
    description = "보통 -- 조정장 매수, 균형 수익/리스크"

    # 매수 조건 (v2: 기준 완화)
    fgi_threshold = 45
    rsi_threshold = 40
    sma_deviation_pct = -2.0
    buy_score_threshold = 50
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
                    if dca_amount < 5000:  # Upbit minimum order is 5000 KRW
                        return Decision(decision="hold", reason="DCA 금액 부족 (최소 5000원 미만)", confidence=0.3, buy_score=buy_score, trade_params={}, external_signal=external_signal, agent_name=f"{self.emoji} {self.name}")
                    return Decision(
                        decision="buy",
                        confidence=0.6,
                        reason=sell_eval["reason"],
                        buy_score=buy_score,
                        trade_params={"side": "bid", "market": "KRW-BTC", "amount": dca_amount, "is_dca": True},
                        external_signal=external_signal,
                        agent_name=f"{self.emoji} {self.name}",
                    )
                elif action == "hold_defer":
                    return Decision(
                        decision="hold",
                        reason=sell_eval.get("reason", "수익 실현 보류 (AI 신호 강세)"),
                        confidence=0.5,
                        buy_score=buy_score,
                        trade_params={},
                        external_signal=external_signal,
                        agent_name=f"{self.emoji} {self.name}",
                    )

        # 매수 판단
        if buy_score["result"] == "buy":
            if ai_score < 0 and fgi > 20:
                d = Decision(
                    decision="hold",
                    confidence=0.5,
                    reason=f"매수 점수 {buy_score['total']}점 충족이나 AI 시그널 음수({ai_score}) → 보류",
                    buy_score=buy_score,
                    trade_params={},
                    external_signal=external_signal,
                    agent_name=f"{self.emoji} {self.name}",
                )
                d._was_ai_vetoed = True
                d._ai_veto_reason = f"ai_signal_negative({ai_score})"
                d._original_action = "buy"
                return d

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

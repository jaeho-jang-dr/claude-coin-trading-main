"""
🛡️ 보수적 전략 에이전트

폭락장 저점 매수만. 발동 빈도 낮지만 자산 보전 우선.
매수 임계: 70점, 손절: -5%, 강제 손절: -10%
"""

from __future__ import annotations

from agents.base_agent import BaseStrategyAgent, Decision


class ConservativeAgent(BaseStrategyAgent):

    name = "conservative"
    emoji = "🛡️"
    description = "보수적 -- 폭락장 저점 매수, 자산 보전 우선"

    # 매수 조건
    fgi_threshold = 30
    rsi_threshold = 30
    sma_deviation_pct = -5.0
    buy_score_threshold = 70
    macd_bonus = False

    # 매도 조건
    target_profit_pct = 15.0
    stop_loss_pct = -5.0
    forced_stop_loss_pct = -10.0
    sell_fgi_threshold = 75
    sell_rsi_threshold = 70

    # 매매 규모
    max_trade_ratio = 0.10
    max_daily_trades = 3
    weekend_reduction = 0.50

    def decide(self, market_data: dict, external_signal: dict, portfolio: dict,
               drop_context: dict | None = None) -> Decision:
        ind = self._extract_indicators(market_data)
        external_bonus = external_signal.get("strategy_bonus", 0)
        ai_score = market_data.get("ai_composite_signal", {}).get("score", 0)
        fgi = market_data.get("fear_greed", {}).get("value", 50)

        # 뉴스 감성 판단
        news = market_data.get("news", {})
        news_negative = news.get("overall_sentiment", "neutral") == "negative"

        # 매수 점수 계산
        buy_score = self.calculate_buy_score(
            fgi=fgi,
            rsi=ind["rsi"],
            sma_deviation=ind["sma_deviation"],
            news_negative=news_negative,
            external_bonus=external_bonus,
        )

        # 보유 중이면 매도 조건 먼저 체크
        btc_holding = portfolio.get("btc", {})
        if btc_holding.get("balance", 0) > 0:
            profit_pct = btc_holding.get("profit_pct", 0)
            # BTC 포지션 비율 계산
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
                        return Decision(decision="hold", reason="DCA 금액 부족 (최소 5000원 미만)", confidence=0.3, trade_params=None)
                    return Decision(
                        decision="buy",
                        confidence=0.6,
                        reason=sell_eval["reason"],
                        buy_score=buy_score,
                        trade_params={
                            "side": "bid",
                            "market": "KRW-BTC",
                            "amount": dca_amount,
                            "is_dca": True,
                        },
                        external_signal=external_signal,
                        agent_name=f"{self.emoji} {self.name}",
                    )
                elif action == "hold_defer":
                    return Decision(
                        decision="hold",
                        reason=sell_eval.get("reason", "수익 실현 보류 (AI 신호 강세)"),
                        confidence=0.5,
                        trade_params=None
                    )

        # 매수 판단
        if buy_score["result"] == "buy":
            # AI 복합 시그널 보조 필터
            if ai_score < 0:
                # 예외: FGI ≤ 15 극단적 공포면 허용
                if fgi <= 15:
                    reason = (f"매수 점수 {buy_score['total']}점 충족. "
                              f"AI 시그널 음수({ai_score})이나 극단 공포(FGI {fgi}) → 매수 허용")
                else:
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
            else:
                reason = f"매수 점수 {buy_score['total']}점 >= {self.buy_score_threshold}점 충족"

            total_krw = portfolio.get("krw_balance", 0)
            amount = self._calculate_trade_amount(total_krw, external_bonus)

            return Decision(
                decision="buy",
                confidence=min(0.9, buy_score["total"] / 100),
                reason=reason,
                buy_score=buy_score,
                trade_params={
                    "side": "bid",
                    "market": "KRW-BTC",
                    "amount": amount,
                },
                external_signal=external_signal,
                agent_name=f"{self.emoji} {self.name}",
            )

        # 관망
        return Decision(
            decision="hold",
            confidence=0.7,
            reason=f"매수 점수 {buy_score['total']}점 < {self.buy_score_threshold}점. 관망.",
            buy_score=buy_score,
            trade_params={},
            external_signal=external_signal,
            agent_name=f"{self.emoji} {self.name}",
        )

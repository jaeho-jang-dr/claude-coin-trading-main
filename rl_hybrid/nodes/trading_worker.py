"""트레이딩 워커 노드 — 매매 실행 + 포트폴리오 관리 전담

역할:
  - Main Brain의 execute_trade 요청 → 안전장치 검증 → Upbit API 매매 실행
  - get_portfolio 요청 → 포트폴리오 조회
  - safety_check 요청 → 안전장치 파라미터 검증
  - PUB 채널 구독하여 최신 시세 캐시
"""

import json
import logging
import os
import sys
import time
from typing import Optional

import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_hybrid.nodes.base_node import BaseNode
from rl_hybrid.config import config
from rl_hybrid.protocol import (
    ZMQMessage, MsgType, Action, make_heartbeat,
)

logger = logging.getLogger("node.trading_worker")


class TradingWorkerNode(BaseNode):
    """트레이딩 워커 — DEALER + SUB"""

    def __init__(self):
        super().__init__("trading_worker")
        self.dealer: Optional[zmq.Socket] = None
        self.sub: Optional[zmq.Socket] = None
        self.latest_market_data: dict = {}

    def _setup_sockets(self):
        # DEALER → Main Brain ROUTER
        self.dealer = self.ctx.socket(zmq.DEALER)
        self.dealer.setsockopt_string(zmq.IDENTITY, self.node_name)
        self.dealer.connect(self.zmq_config.router_addr)
        self.logger.info(f"DEALER 연결: {self.zmq_config.router_addr}")

        # SUB → Main Brain PUB
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(self.zmq_config.pub_addr)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.logger.info(f"SUB 연결: {self.zmq_config.pub_addr}")

    def _send_heartbeat(self):
        """Main Brain에 하트비트 전송"""
        msg = make_heartbeat(self.node_name, status="alive", uptime=self.uptime)
        try:
            self.dealer.send(b"" + msg.serialize())
        except Exception:
            pass

    def _main_loop(self):
        """메시지 수신 루프"""
        poller = zmq.Poller()
        poller.register(self.dealer, zmq.POLLIN)
        poller.register(self.sub, zmq.POLLIN)

        self.logger.info("Trading Worker 메인 루프 시작")

        while self._running:
            events = dict(poller.poll(100))

            if self.dealer in events:
                self._handle_dealer_message()

            if self.sub in events:
                self._handle_sub_message()

    def _handle_dealer_message(self):
        """Main Brain 요청 처리"""
        try:
            frames = self.dealer.recv_multipart(zmq.NOBLOCK)
            data = frames[-1] if frames else None
            if not data:
                return
            msg = ZMQMessage.deserialize(data)
        except Exception as e:
            self.logger.error(f"메시지 수신 에러: {e}")
            return

        self.logger.info(f"요청 수신: action={msg.action}, req_id={msg.request_id}")

        handlers = {
            Action.EXECUTE_TRADE.value: self._handle_execute_trade,
            Action.GET_PORTFOLIO.value: self._handle_get_portfolio,
            Action.SAFETY_CHECK.value: self._handle_safety_check,
        }

        handler = handlers.get(msg.action)
        if handler:
            response = handler(msg)
            if response:
                response.sender = self.node_name
                self.dealer.send(b"" + response.serialize())
        else:
            self.logger.warning(f"알 수 없는 액션: {msg.action}")

    def _handle_sub_message(self):
        """PUB 채널 시장 데이터 캐시"""
        try:
            data = self.sub.recv(zmq.NOBLOCK)
            msg = ZMQMessage.deserialize(data)
            if msg.action == Action.MARKET_UPDATE.value:
                self.latest_market_data = msg.payload
        except Exception:
            pass

    def _handle_execute_trade(self, msg: ZMQMessage) -> Optional[ZMQMessage]:
        """매매 실행 요청 처리"""
        side = msg.payload.get("side")  # "buy" or "sell"
        market = msg.payload.get("market", "KRW-BTC")
        amount = msg.payload.get("amount")
        volume = msg.payload.get("volume")
        reason = msg.payload.get("reason", "")

        # 1. 안전장치 검증
        safety = self._check_safety(side, amount)
        if not safety["safe"]:
            self.logger.warning(f"안전장치 차단: {safety['reason']}")
            return msg.reply({
                "executed": False,
                "reason": safety["reason"],
            })

        # 2. 매매 실행 (기존 execute_trade.py 호출)
        try:
            import subprocess
            from scripts.hide_console import subprocess_kwargs
            cmd_args = [
                sys.executable, "scripts/execute_trade.py",
                "--side", side,
                "--market", market,
            ]
            if amount:
                cmd_args.extend(["--amount", str(amount)])
            if volume:
                cmd_args.extend(["--volume", str(volume)])

            result = subprocess.run(
                cmd_args,
                capture_output=True, text=True, timeout=30,
                cwd=config.project_root,
                **subprocess_kwargs(),
            )

            if result.returncode == 0:
                trade_result = json.loads(result.stdout)
                self.logger.info(f"매매 체결: {side} {market} {amount or volume}")
                return msg.reply({
                    "executed": True,
                    "result": trade_result,
                })
            else:
                self.logger.error(f"매매 실행 실패: {result.stderr[:500]}")
                return msg.reply({
                    "executed": False,
                    "reason": result.stderr[:500],
                })

        except Exception as e:
            self.logger.error(f"매매 실행 에러: {e}")
            return msg.reply({"executed": False, "reason": str(e)})

    def _handle_get_portfolio(self, msg: ZMQMessage) -> Optional[ZMQMessage]:
        """포트폴리오 조회"""
        try:
            import subprocess
            from scripts.hide_console import subprocess_kwargs
            result = subprocess.run(
                [sys.executable, "scripts/get_portfolio.py"],
                capture_output=True, text=True, timeout=15,
                cwd=config.project_root,
                **subprocess_kwargs(),
            )

            if result.returncode == 0:
                portfolio = json.loads(result.stdout)
                return msg.reply(portfolio)
            else:
                return msg.reply({}, error=result.stderr[:500])

        except Exception as e:
            return msg.reply({}, error=str(e))

    def _handle_safety_check(self, msg: ZMQMessage) -> Optional[ZMQMessage]:
        """안전장치 검증"""
        action = msg.payload.get("action", "buy")
        params = msg.payload.get("params", {})
        safety = self._check_safety(action, params.get("amount"))
        return msg.reply(safety)

    def _check_safety(self, side: str, amount: float = None) -> dict:
        """안전장치 종합 검증"""
        from dotenv import load_dotenv
        load_dotenv(override=True)

        # EMERGENCY_STOP 확인
        if os.getenv("EMERGENCY_STOP", "false").lower() == "true":
            return {"safe": False, "reason": "EMERGENCY_STOP 활성화"}

        # auto_emergency.json 확인
        auto_emergency_path = os.path.join(config.project_root, "data", "auto_emergency.json")
        if os.path.exists(auto_emergency_path):
            try:
                with open(auto_emergency_path, "r") as f:
                    ae = json.load(f)
                if ae.get("active", False) and side == "buy":
                    return {"safe": False, "reason": f"자동 긴급정지 활성: {ae.get('reason', '')}"}
            except Exception:
                pass

        # DRY_RUN 확인
        if os.getenv("DRY_RUN", "true").lower() == "true":
            return {"safe": False, "reason": "DRY_RUN 모드 — 실제 매매 비활성"}

        # MAX_TRADE_AMOUNT 확인
        max_amount = int(os.getenv("MAX_TRADE_AMOUNT", "100000"))
        if amount and float(amount) > max_amount:
            return {"safe": False, "reason": f"MAX_TRADE_AMOUNT 초과: {amount} > {max_amount}"}

        return {"safe": True, "reason": "OK"}


if __name__ == "__main__":
    node = TradingWorkerNode()
    node.start()

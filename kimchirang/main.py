"""Kimchirang Main -- 진입점 + RL 브릿지 + 메인 이벤트 루프

실행:
  python -m kimchirang.main              # 기본 (DRY_RUN=true)
  KR_DRY_RUN=false python -m kimchirang.main  # 실매매
"""

import asyncio
import logging
import os
import signal
import sys
import time

import numpy as np

# 프로젝트 루트를 path에 추가
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from kimchirang.config import KimchirangConfig
from kimchirang.data_feeder import DataFeeder
from kimchirang.kp_engine import KPEngine
from kimchirang.execution import Executor
from kimchirang.notifier import KimchirangNotifier
from kimchirang.db import KimchirangDB

logger = logging.getLogger("kimchirang.main")

# RL 액션 코드
ACTION_HOLD = 0
ACTION_ENTER = 1
ACTION_EXIT = 2


# ============================================================
# RL 에이전트 브릿지
# ============================================================

class RLBridge:
    """PPO + DQN 앙상블 RL 브릿지

    두 모델이 모두 있으면 앙상블 (합의 시만 행동),
    하나만 있으면 단독 사용, 없으면 규칙 기반 fallback.
    """

    def __init__(self, config: KimchirangConfig):
        self.config = config
        self._ppo_model = None
        self._dqn_model = None
        self._available = False

    def load(self) -> bool:
        """RL 모델 로드 시도 (PPO + DQN)"""
        if not self.config.rl.enabled:
            logger.info("RL 비활성화 -- 규칙 기반 모드")
            return False

        loaded = 0

        # PPO 로드
        ppo_path = os.path.join(PROJECT_DIR, self.config.rl.model_path)
        try:
            from stable_baselines3 import PPO
            if os.path.exists(ppo_path + ".zip") or os.path.exists(ppo_path):
                self._ppo_model = PPO.load(ppo_path)
                loaded += 1
                logger.info(f"PPO 모델 로드 완료: {ppo_path}")
        except Exception as e:
            logger.warning(f"PPO 모델 로드 실패: {e}")

        # DQN 로드
        dqn_path = os.path.join(
            PROJECT_DIR, "data", "rl_models", "kimchirang", "dqn", "best_model"
        )
        try:
            from stable_baselines3 import DQN
            if os.path.exists(dqn_path + ".zip") or os.path.exists(dqn_path):
                self._dqn_model = DQN.load(dqn_path)
                loaded += 1
                logger.info(f"DQN 모델 로드 완료: {dqn_path}")
        except Exception as e:
            logger.warning(f"DQN 모델 로드 실패: {e}")

        self._available = loaded > 0
        if self._ppo_model and self._dqn_model:
            logger.info("앙상블 모드: PPO + DQN (합의 시 행동)")
        elif self._available:
            which = "PPO" if self._ppo_model else "DQN"
            logger.info(f"단독 모드: {which}")
        else:
            logger.warning("RL 모델 없음 -- 규칙 기반 모드")

        return self._available

    def get_action(self, state: np.ndarray) -> int:
        """RL 에이전트에게 액션 요청 (앙상블)

        앙상블 규칙:
          - 둘 다 같은 액션 → 그대로 실행
          - 한쪽만 Enter/Exit → Hold (신중하게)
          - 단독 모델이면 그대로 실행

        Returns:
            0: Hold, 1: Enter, 2: Exit
        """
        if not self._available:
            return ACTION_HOLD

        ppo_action = self._predict(self._ppo_model, state, "PPO")
        dqn_action = self._predict(self._dqn_model, state, "DQN")

        # 단독 모델
        if ppo_action is not None and dqn_action is None:
            return ppo_action
        if dqn_action is not None and ppo_action is None:
            return dqn_action

        # 앙상블: 합의
        if ppo_action == dqn_action:
            return ppo_action

        # 불일치 → Hold (신중)
        return ACTION_HOLD

    def _predict(self, model, state: np.ndarray, name: str):
        """단일 모델 추론"""
        if model is None:
            return None
        try:
            action, _ = model.predict(state, deterministic=True)
            return int(action)
        except Exception as e:
            logger.error(f"{name} 추론 실패: {e}")
            return None

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def mode(self) -> str:
        if self._ppo_model and self._dqn_model:
            return "ensemble"
        elif self._ppo_model:
            return "ppo"
        elif self._dqn_model:
            return "dqn"
        return "rule"


# ============================================================
# 메인 트레이딩 루프
# ============================================================

class KimchirangBot:
    """김치프리미엄 차익거래 봇 메인 클래스"""

    def __init__(self):
        self.config = KimchirangConfig()
        self.feeder = DataFeeder(self.config)
        self.engine = KPEngine(self.config, self.feeder.state)
        self.executor = Executor(self.config)
        self.rl_bridge = RLBridge(self.config)
        self.notifier = KimchirangNotifier()
        self.db = KimchirangDB(self.config.db)
        self._running = False
        self._tick_count = 0

    async def run(self):
        """메인 실행"""
        self._running = True

        # 설정 검증
        errors = self.config.validate()
        if errors and not self.config.trading.dry_run:
            for e in errors:
                logger.error(f"설정 오류: {e}")
            logger.error("실매매 모드에서 설정 오류 -- 중단")
            return

        logger.info(self.config.summary())

        # RL 모델 로드
        self.rl_bridge.load()

        # Binance 레버리지 설정
        lev_ok = await self.executor.set_leverage()
        if not lev_ok and not self.config.trading.dry_run:
            logger.error("Binance 레버리지 설정 실패 -- 실매매 모드에서 중단")
            return

        # 데이터 피더 시작
        await self.feeder.start()

        # 데이터 준비 대기
        ready = await self.feeder.wait_ready(timeout=60)
        if not ready:
            logger.error("데이터 피드 준비 실패 -- 중단")
            await self.feeder.stop()
            return

        logger.info("=== Kimchirang 봇 시작 ===")

        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("봇 중단 요청")
        finally:
            await self._shutdown()

    async def _main_loop(self):
        """메인 트레이딩 루프 (1초 간격)"""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"Tick 오류: {e}", exc_info=True)

            await asyncio.sleep(1.0)

    async def _tick(self):
        """1초마다 실행되는 메인 로직"""
        self._tick_count += 1

        # 1. KP 계산
        snapshot = self.engine.calculate()
        if not snapshot.is_valid:
            return

        stats = self.engine.get_stats()
        pos = self.executor.get_position_info()

        # 2. 주기적 상태 로그 (10초마다)
        if self._tick_count % 10 == 0:
            logger.info(
                f"[KP] mid={stats['mid_kp']:+.2f}% "
                f"entry={stats['entry_kp']:+.2f}% "
                f"exit={stats['exit_kp']:+.2f}% "
                f"| MA5m={stats['kp_ma_5m']:+.2f}% "
                f"Z={stats['kp_z_score']:+.2f} "
                f"V={stats['kp_velocity']:+.3f}%/m "
                f"| FR={stats['funding_rate']*100:+.4f}% "
                f"| pos={pos['side']}"
            )

        # 2b. KP 히스토리 DB 기록 (1분마다)
        if self._tick_count % 60 == 0:
            await self.db.record_kp_snapshot(snapshot, stats)

        # 2c. 주기적 텔레그램 상태 보고 (5분마다)
        if self._tick_count % 300 == 0:
            await self.notifier.notify_status(stats, pos)

        # 3. 데이터 신선도 체크
        age = self.feeder.state.data_age_sec
        if any(v > 10 for v in age.values() if v >= 0):
            if self._tick_count % 30 == 0:
                logger.warning(f"데이터 지연: {age}")
            return

        # 4. RL 에이전트 액션 결정
        action = self._decide_action(snapshot)

        # 5. 주문 실행
        if action == ACTION_ENTER:
            result = await self.executor.enter(snapshot)
            if result.both_success:
                await self.notifier.notify_enter(result, snapshot)
                await self.db.record_trade(result, snapshot, stats=stats)

        elif action == ACTION_EXIT:
            hold_min = self.executor.position.hold_duration_min
            result = await self.executor.exit(snapshot)
            if result.both_success:
                pnl = self.executor._calculate_pnl(result, snapshot)
                await self.notifier.notify_exit(result, snapshot, pnl)
                await self.db.record_trade(
                    result, snapshot, pnl=pnl, stats=stats,
                    hold_duration_min=hold_min,
                )

        # 6. 손절 체크 (포지션 보유 중)
        if self.executor.position.is_open and self.engine.should_stop_loss(snapshot):
            logger.warning(f"손절 트리거: KP={snapshot.mid_kp:.2f}%")
            hold_min = self.executor.position.hold_duration_min
            result = await self.executor.exit(snapshot)
            if result.both_success:
                pnl = self.executor._calculate_pnl(result, snapshot)
                await self.notifier.notify_stop_loss(result, snapshot, pnl)
                await self.db.record_trade(
                    result, snapshot, pnl=pnl, stats=stats,
                    hold_duration_min=hold_min,
                )

    def _decide_action(self, snapshot) -> int:
        """RL + 규칙 기반 하이브리드 결정"""
        # RL 에이전트 사용 가능하면 RL 우선
        if self.rl_bridge.is_available:
            state = self.engine.build_rl_state()
            # 포지션 상태를 state에 주입
            state[11] = 1.0 if self.executor.position.is_open else 0.0
            rl_action = self.rl_bridge.get_action(state)

            # RL 액션을 규칙 기반으로 검증 (안전장치)
            if rl_action == ACTION_ENTER and not self.engine.should_enter(snapshot):
                # RL이 진입 원하지만 KP가 임계값 미만이면 차단
                if snapshot.entry_kp < self.config.trading.kp_entry_threshold * 0.7:
                    return ACTION_HOLD
            return rl_action

        # 규칙 기반 fallback
        if not self.executor.position.is_open:
            if self.engine.should_enter(snapshot):
                return ACTION_ENTER
        else:
            if self.engine.should_exit(snapshot):
                return ACTION_EXIT

        return ACTION_HOLD

    async def _shutdown(self):
        """정리 종료"""
        logger.info("Kimchirang 종료 중...")
        self._running = False
        # 포지션 상태 최종 저장
        self.executor._save_state()
        await self.executor.close()
        await self.feeder.stop()
        logger.info("Kimchirang 종료 완료")

    def stop(self):
        self._running = False


# ============================================================
# 진입점
# ============================================================

def setup_logging():
    log_dir = os.path.join(PROJECT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, "kimchirang.log"),
                encoding="utf-8",
            ),
        ],
    )


def main():
    setup_logging()
    logger.info("Kimchirang v0.1.0 시작")

    bot = KimchirangBot()

    # Graceful shutdown 핸들러
    def handle_signal(*_):
        logger.info("종료 시그널 수신")
        bot.stop()

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except (OSError, ValueError):
        pass  # Windows에서 일부 시그널 미지원

    asyncio.run(bot.run())


if __name__ == "__main__":
    main()

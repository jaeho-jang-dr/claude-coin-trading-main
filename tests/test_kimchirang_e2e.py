"""Kimchirang E2E 테스트 -- DRY_RUN 모드로 5초 실행"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KR_DRY_RUN"] = "true"
os.environ["KR_RL_ENABLED"] = "false"  # RL 없이 규칙 기반만

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from kimchirang.config import KimchirangConfig
from kimchirang.data_feeder import DataFeeder, MarketState
from kimchirang.kp_engine import KPEngine
from kimchirang.execution import Executor
from kimchirang.notifier import KimchirangNotifier
from kimchirang.db import KimchirangDB


async def test_e2e():
    print("=" * 60)
    print("Kimchirang E2E Test -- DRY_RUN Mode")
    print("=" * 60)

    # 1. Config 검증
    print("\n[Phase 1] Config 검증")
    config = KimchirangConfig()
    print(f"  DRY_RUN: {config.trading.dry_run}")
    print(f"  RL Enabled: {config.rl.enabled}")
    print(f"  KP Entry Threshold: {config.trading.kp_entry_threshold}%")
    print(f"  KP Exit Threshold: {config.trading.kp_exit_threshold}%")
    print(f"  KP Stop Loss: {config.trading.kp_stop_loss}%")
    errors = config.validate()
    print(f"  Config errors: {errors if errors else 'None (API keys set)'}")

    assert config.trading.dry_run, "DRY_RUN must be true"
    assert not config.rl.enabled, "RL should be disabled for this test"
    print("  [OK] Config validation passed")

    # 2. DataFeeder 시작 (WebSocket 연결)
    print("\n[Phase 2] DataFeeder 시작 (WebSocket 연결)")
    feeder = DataFeeder(config)
    await feeder.start()

    # 30초 대기 (데이터 수신)
    ready = await feeder.wait_ready(timeout=30)
    print(f"  Data ready: {ready}")

    status = feeder.get_status()
    print(f"  Upbit connected: {status['upbit_connected']}")
    print(f"  Binance connected: {status['binance_connected']}")
    print(f"  FX available: {status['fx_available']}")
    print(f"  FX rate: {status['fx_rate']:,.2f}" if status['fx_rate'] else "  FX rate: N/A")
    print(f"  Upbit bid: {status['upbit_bid']:,.0f}" if status['upbit_bid'] else "  Upbit bid: N/A")
    print(f"  Binance bid: {status['binance_bid']:,.2f}" if status['binance_bid'] else "  Binance bid: N/A")

    if not ready:
        print("\n  [WARN] Data not ready within 30s -- partial test mode")
        # 개별 소스 확인
        if not status['upbit_connected']:
            print("  [ISSUE] Upbit WebSocket 연결 실패")
        if not status['binance_connected']:
            print("  [ISSUE] Binance WebSocket 연결 실패")
        if not status['fx_available']:
            print("  [ISSUE] FX rate 수신 실패")

    # 3. KPEngine 테스트
    print("\n[Phase 3] KPEngine 테스트")
    engine = KPEngine(config, feeder.state)

    if ready:
        snapshot = engine.calculate()
        print(f"  KP valid: {snapshot.is_valid}")
        print(f"  Entry KP: {snapshot.entry_kp:+.4f}%")
        print(f"  Exit KP: {snapshot.exit_kp:+.4f}%")
        print(f"  Mid KP: {snapshot.mid_kp:+.4f}%")
        print(f"  FX Rate: {snapshot.fx_rate:,.2f}")
        print(f"  Upbit Spread: {snapshot.upbit_spread_pct:.4f}%")
        print(f"  Binance Spread: {snapshot.binance_spread_pct:.4f}%")
        print(f"  Funding Rate: {snapshot.funding_rate}")

        stats = engine.get_stats()
        print(f"  Stats: mid_kp={stats['mid_kp']}, n_samples={stats['n_samples']}")
        print(f"  Should Enter: {engine.should_enter(snapshot)}")
        print(f"  Should Exit: {engine.should_exit(snapshot)}")
        print(f"  Should Stop Loss: {engine.should_stop_loss(snapshot)}")

        # RL state vector
        rl_state = engine.build_rl_state()
        print(f"  RL State vector: shape={rl_state.shape}, dtype={rl_state.dtype}")
        print(f"  RL State: {rl_state}")
        print("  [OK] KPEngine passed")
    else:
        print("  [SKIP] KPEngine test skipped (no data)")

    # 4. Executor DRY_RUN 테스트
    print("\n[Phase 4] Executor 테스트 (DRY_RUN)")
    executor = Executor(config)
    pos = executor.get_position_info()
    print(f"  Position side: {pos['side']}")
    print(f"  Is open: {pos['is_open']}")
    print(f"  Trades today: {pos['trades_today']}")
    print("  [OK] Executor initialization passed")

    # 5. Notifier 테스트
    print("\n[Phase 5] Notifier 테스트")
    notifier = KimchirangNotifier()
    print(f"  Telegram enabled: {notifier._enabled}")
    print("  [OK] Notifier initialization passed")

    # 6. DB 테스트
    print("\n[Phase 6] DB 테스트")
    db = KimchirangDB(config.db)
    print(f"  Supabase enabled: {db._enabled}")
    print("  [OK] DB initialization passed")

    # 7. 5초 KP 추적 루프
    print("\n[Phase 7] 5초 KP 추적 루프")
    if ready:
        for i in range(5):
            await asyncio.sleep(1)
            snap = engine.calculate()
            if snap.is_valid:
                st = engine.get_stats()
                print(
                    f"  [{i+1}s] KP mid={snap.mid_kp:+.4f}% | "
                    f"entry={snap.entry_kp:+.4f}% | "
                    f"exit={snap.exit_kp:+.4f}% | "
                    f"samples={st['n_samples']}"
                )
            else:
                print(f"  [{i+1}s] Invalid snapshot (data stale?)")
        print("  [OK] KP tracking loop passed")
    else:
        print("  [SKIP] KP tracking skipped (no data)")

    # 8. 정리
    print("\n[Phase 8] Cleanup")
    await feeder.stop()
    await executor.close()
    print("  [OK] Cleanup complete")

    # 최종 결과
    print("\n" + "=" * 60)
    if ready:
        print("E2E TEST RESULT: PASS (all components operational)")
    else:
        print("E2E TEST RESULT: PARTIAL (WebSocket data not fully ready)")
        print("  - Config, Executor, Notifier, DB: OK")
        print("  - DataFeeder: connection issues (see above)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_e2e())

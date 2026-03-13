"""Kimchirang 통합 E2E -- 텔레그램 알림 + Supabase DB + 로컬 JSONL"""
import asyncio, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kimchirang.config import KimchirangConfig, DBConfig
from kimchirang.db import KimchirangDB
from kimchirang.notifier import KimchirangNotifier
from kimchirang.execution import ExecutionResult, LegResult
from kimchirang.kp_engine import KPSnapshot
from kimchirang.state import load_position, save_position

async def test_integrations():
    config = KimchirangConfig()

    # 1. 텔레그램 상태 알림 테스트
    print("=== 텔레그램 테스트 ===")
    notifier = KimchirangNotifier()
    await notifier.notify_status(
        {"mid_kp": 1.5, "kp_z_score": 0.3, "kp_velocity": 0.01},
        {"side": "none", "is_open": False, "trades_today": 0}
    )
    print("텔레그램 알림 전송 완료")

    # 2. DB 기록 테스트 (로컬 JSONL)
    print("\n=== DB 기록 테스트 ===")
    db = KimchirangDB(config.db)
    snapshot = KPSnapshot(
        entry_kp=3.5, exit_kp=0.3, mid_kp=1.9,
        upbit_bid=85000000, upbit_ask=85100000,
        binance_bid=62000, binance_ask=62050,
        fx_rate=1350, upbit_spread_pct=0.12,
        binance_spread_pct=0.08, funding_rate=0.0001,
        timestamp=time.time()
    )
    result = ExecutionResult(
        action="enter", kp_at_execution=3.5, dry_run=True,
        upbit_leg=LegResult(exchange="upbit", success=True, filled_price=85100000, filled_qty=0.001),
        binance_leg=LegResult(exchange="binance", success=True, filled_price=62000, filled_qty=0.001),
    )
    await db.record_trade(result, snapshot, stats={"mid_kp": 1.9})
    await db.record_kp_snapshot(snapshot, {"mid_kp": 1.9, "kp_ma_1m": 1.8, "kp_ma_5m": 1.7, "kp_z_score": 0.3, "kp_velocity": 0.01, "spread_cost": 0.2, "funding_rate": 0.0001})

    # 로컬 JSONL 확인
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kimchirang")
    for f in ["trades.jsonl", "kp_history.jsonl"]:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
            print(f"  {f}: {len(lines)}줄")
            # 마지막 줄 파싱 검증
            last = json.loads(lines[-1])
            print(f"    마지막 기록: {list(last.keys())[:5]}...")
        else:
            print(f"  {f}: 없음!")

    # 3. 포지션 영속성 테스트
    print("\n=== 포지션 영속성 테스트 ===")
    test_state = {"side": "long_kp", "entry_kp": 3.5, "entry_time": time.time(),
                  "upbit_qty": 0.001, "binance_qty": 0.001,
                  "upbit_entry_price": 85000000, "binance_entry_price": 62000,
                  "trade_count_today": 1, "last_trade_time": time.time()}
    save_position(test_state)
    loaded = load_position()
    assert loaded["side"] == "long_kp", f"Expected long_kp, got {loaded['side']}"
    # 원복
    save_position({"side": "none", "entry_kp": 0, "entry_time": 0,
                   "upbit_qty": 0, "binance_qty": 0,
                   "upbit_entry_price": 0, "binance_entry_price": 0,
                   "trade_count_today": 0, "last_trade_time": 0})
    print("포지션 저장/복원 OK")

    print("\n통합 E2E 테스트 완료!")

asyncio.run(test_integrations())

"""전체 노드 런처 — 4개 노드 + 지속 학습 + 주간 재학습 + 월간 고급 훈련

단일 머신 테스트용. 분산 배포 시에는 각 start_*.py를 개별 머신에서 실행.

사용법:
    python start_all.py                    # 전체 (4노드 + 지속학습 + 주간재학습)
    python start_all.py --no-rl            # RL Worker 제외 (3노드)
    python start_all.py --rl-workers 3     # RL Worker 3개 기동
    python start_all.py --no-weekly        # 주간 재학습 스케줄러 제외
    python start_all.py --with-multi-agent # 월간 Multi-Agent 훈련 추가
    python start_all.py --with-dt-retrain  # 월간 Decision Transformer 재훈련 추가
    python start_all.py --with-offline-rl  # 월간 Offline RL 훈련 추가
"""

import argparse
import os
import subprocess
from scripts.hide_console import subprocess_kwargs
import sys
import threading
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYTHON = sys.executable
RL_HYBRID = os.path.join(PROJECT_ROOT, "rl_hybrid")

sys.path.insert(0, PROJECT_ROOT)

processes = []


def start_node(name: str, script: str, extra_args: list = None):
    """단일 노드 시작"""
    log_dir = os.path.join(PROJECT_ROOT, "logs", "nodes")
    os.makedirs(log_dir, exist_ok=True)

    log_file = open(os.path.join(log_dir, f"{name}_stdout.log"), "a", encoding="utf-8")
    cmd = [PYTHON, script] + (extra_args or [])

    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        **subprocess_kwargs(),
    )
    processes.append((name, proc, log_file))
    print(f"  [{name}] PID={proc.pid}")
    return proc


def start_continuous_learning():
    """Track A: 지속 학습 백그라운드 시작 (6시간 간격)"""
    print("  [continuous_learning] 시작 중...")
    start_node(
        "continuous_learning",
        os.path.join(RL_HYBRID, "launchers", "start_continuous_learning.py"),
        ["--interval", "6", "--steps", "20000", "--days", "30"],
    )


def start_weekly_retrain_scheduler():
    """Track B: 주간 재학습 스케줄러 (일요일 03:00)

    Windows 작업 스케줄러 대신 프로세스 내 스레드로 관리.
    start_all.py가 항상 실행 중이므로 별도 스케줄러 불필요.
    """
    def _weekly_loop():
        import logging
        logger = logging.getLogger("weekly_scheduler")
        logger.info("주간 재학습 스케줄러 시작 (일요일 03:00)")

        while True:
            now = datetime.now()
            # 일요일(6) 03:00 체크
            if now.weekday() == 6 and now.hour == 3 and now.minute == 0:
                logger.info("=== 주간 심층 재학습 트리거 ===")
                try:
                    result = subprocess.run(
                        [PYTHON, os.path.join(PROJECT_ROOT, "scripts", "weekly_retrain.py")],
                        cwd=PROJECT_ROOT,
                        capture_output=True, text=True,
                        timeout=1800,  # 최대 30분
                        **subprocess_kwargs(),
                    )
                    if result.returncode == 0:
                        logger.info("주간 재학습 완료")
                    else:
                        logger.error(f"주간 재학습 실패: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    logger.error("주간 재학습 타임아웃 (30분)")
                except Exception as e:
                    logger.error(f"주간 재학습 에러: {e}")

                # 같은 시간 중복 실행 방지 (61초 대기)
                time.sleep(61)
            else:
                time.sleep(30)  # 30초마다 체크

    thread = threading.Thread(target=_weekly_loop, daemon=True, name="weekly_retrain")
    thread.start()
    print(f"  [weekly_retrain] 스케줄러 시작 (일요일 03:00)")


def start_monthly_advanced_training(
    with_multi_agent: bool = False,
    with_dt_retrain: bool = False,
    with_offline_rl: bool = False,
    stop_event: threading.Event = None,
):
    """월간 고급 모델 훈련 (매월 1일 04:00)

    선택적으로 Multi-Agent, Decision Transformer, Offline RL 재훈련을
    월 1회 백그라운드에서 실행한다.
    """
    import logging
    logger = logging.getLogger("monthly_advanced")
    _stop = stop_event or threading.Event()

    components = []
    if with_multi_agent:
        components.append("Multi-Agent")
    if with_dt_retrain:
        components.append("Decision Transformer")
    if with_offline_rl:
        components.append("Offline RL")

    logger.info(f"월간 고급 훈련 스케줄러 시작 (매월 1일 04:00): {', '.join(components)}")

    def _monthly_loop():
        while not _stop.is_set():
            now = datetime.now()
            # 매월 1일 04:00~04:09
            if now.day == 1 and now.hour == 4 and now.minute < 10:
                logger.info("=== 월간 고급 모델 훈련 트리거 ===")

                # Multi-Agent 훈련
                if with_multi_agent:
                    try:
                        from rl_hybrid.rl.multi_agent_consensus import MultiAgentTrainer
                        trainer = MultiAgentTrainer()
                        trainer.train(scalping_days=90, swing_days=180)
                        logger.info("Multi-Agent 훈련 완료")
                    except Exception as e:
                        logger.error(f"Multi-Agent 훈련 실패: {e}")

                # Decision Transformer 재훈련
                if with_dt_retrain:
                    try:
                        from rl_hybrid.rl.decision_transformer import train_dt
                        result = train_dt(days=180, interval="4h", n_epochs=100)
                        logger.info(f"Decision Transformer 재훈련 완료: {result.get('status', 'unknown')}")
                    except Exception as e:
                        logger.error(f"Decision Transformer 재훈련 실패: {e}")

                # Offline RL 훈련
                if with_offline_rl:
                    try:
                        from rl_hybrid.rl.offline_rl import train_offline
                        result = train_offline(algorithm="cql", epochs=100)
                        logger.info(f"Offline RL 훈련 완료: {result.get('status', 'unknown')}")
                    except Exception as e:
                        logger.error(f"Offline RL 훈련 실패: {e}")

                # 같은 시간 중복 실행 방지 (11분 대기)
                _stop.wait(660)
            else:
                _stop.wait(600)  # 10분마다 체크

    thread = threading.Thread(target=_monthly_loop, daemon=True, name="monthly_advanced")
    thread.start()
    return thread


def start_all(
    n_rl_workers: int = 1,
    no_rl: bool = False,
    no_weekly: bool = False,
    with_multi_agent: bool = False,
    with_dt_retrain: bool = False,
    with_offline_rl: bool = False,
):
    """모든 노드 + 학습 스케줄러 시작"""
    print("=== 분산 LLM-RL 하이브리드 트레이딩 시스템 ===\n")

    # 1. Main Brain (ROUTER + PUB + 글로벌 트레이너)
    print("  [main_brain] 시작 중...")
    start_node("main_brain", os.path.join(RL_HYBRID, "nodes", "main_brain.py"),
               ["--no-rl"] if no_rl else [])
    time.sleep(2)

    # 2. LLM/RAG Worker
    print("  [llm_worker] 시작 중...")
    start_node("llm_worker", os.path.join(RL_HYBRID, "nodes", "llm_worker.py"))
    time.sleep(1)

    # 3. Trading Worker
    print("  [trading_worker] 시작 중...")
    start_node("trading_worker", os.path.join(RL_HYBRID, "nodes", "trading_worker.py"))
    time.sleep(1)

    # 4. RL Workers
    if not no_rl:
        for i in range(n_rl_workers):
            worker_id = f"rl_worker_{i}"
            print(f"  [{worker_id}] 시작 중...")
            start_node(
                worker_id,
                os.path.join(RL_HYBRID, "nodes", "rl_worker.py"),
                ["--id", worker_id],
            )
            time.sleep(1)

    # 5. Track A: 지속 학습 (6시간 간격)
    start_continuous_learning()
    time.sleep(1)

    # 6. Track B: 주간 재학습 스케줄러 (일요일 03:00)
    if not no_weekly:
        start_weekly_retrain_scheduler()

    # 7. 월간 고급 모델 훈련 (매월 1일 04:00)
    has_monthly = with_multi_agent or with_dt_retrain or with_offline_rl
    if has_monthly:
        start_monthly_advanced_training(
            with_multi_agent=with_multi_agent,
            with_dt_retrain=with_dt_retrain,
            with_offline_rl=with_offline_rl,
        )

    total = len(processes)
    print(f"\n전체 {total}개 노드 + 학습 스케줄러 실행 중.")
    print("  - Main Brain:          글로벌 PPO 트레이너 + 오케스트레이션")
    print("  - LLM Worker:          Gemini 분석 + RAG 파이프라인")
    print("  - Trading Worker:      매매 실행 + 안전장치")
    if not no_rl:
        print(f"  - RL Worker x{n_rl_workers}:        로컬 환경 롤아웃 수집")
    print("  - Continuous Learning: Track A — 6시간 증분 학습 (20K 스텝)")
    if not no_weekly:
        print("  - Weekly Retrain:      Track B — 일요일 03:00 심층 학습 (500K 스텝)")
    if has_monthly:
        parts = []
        if with_multi_agent:
            parts.append("Multi-Agent")
        if with_dt_retrain:
            parts.append("DT")
        if with_offline_rl:
            parts.append("Offline RL")
        print(f"  - Monthly Advanced:    매월 1일 04:00 — {', '.join(parts)}")
    print("\n종료: Ctrl+C\n")


def stop_all():
    """모든 노드 종료"""
    print("\n=== 전체 노드 종료 중 ===")
    for name, proc, log_file in reversed(processes):
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print(f"  [{name}] 종료 완료 (PID={proc.pid})")
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"  [{name}] 강제 종료 (PID={proc.pid})")
        finally:
            log_file.close()


def monitor():
    """프로세스 상태 모니터링 + 자동 재시작"""
    while True:
        time.sleep(10)
        for name, proc, _ in processes:
            ret = proc.poll()
            if ret is not None:
                print(f"  [경고] {name} 종료됨 (code={ret})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="분산 LLM-RL 시스템 런처")
    parser.add_argument("--no-rl", action="store_true", help="RL Worker 없이 시작")
    parser.add_argument("--rl-workers", type=int, default=1, help="RL Worker 개수")
    parser.add_argument("--no-weekly", action="store_true", help="주간 재학습 제외")
    parser.add_argument("--with-multi-agent", action="store_true",
                        help="월간 Multi-Agent 합의 RL 훈련 활성화")
    parser.add_argument("--with-dt-retrain", action="store_true",
                        help="월간 Decision Transformer 재훈련 활성화")
    parser.add_argument("--with-offline-rl", action="store_true",
                        help="월간 Offline RL (CQL/BCQ) 훈련 활성화")
    args = parser.parse_args()

    try:
        start_all(
            n_rl_workers=args.rl_workers,
            no_rl=args.no_rl,
            no_weekly=args.no_weekly,
            with_multi_agent=args.with_multi_agent,
            with_dt_retrain=args.with_dt_retrain,
            with_offline_rl=args.with_offline_rl,
        )
        monitor()
    except KeyboardInterrupt:
        stop_all()

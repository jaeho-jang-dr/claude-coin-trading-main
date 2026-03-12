"""Admin 전용 — 훈련 결과 리뷰 + 모델 승격

Tier 3 (Admin)만 사용합니다.
제출된 훈련 결과를 비교하고, 최고 모델을 best로 승격합니다.

사용법:
    python -m rl_hybrid.rl.admin_review --list           # 제출 목록 조회
    python -m rl_hybrid.rl.admin_review --promote ID     # 특정 제출 승격
    python -m rl_hybrid.rl.admin_review --leaderboard    # 리더보드 출력
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rl.admin")


def get_submissions_from_db() -> list[dict]:
    """DB에서 제출 목록 조회"""
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not supabase_url or not supabase_key:
        return get_submissions_local()

    try:
        import requests
        resp = requests.get(
            f"{supabase_url}/rest/v1/rl_training_results"
            "?order=avg_return_pct.desc&limit=50",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"DB 조회 실패: {e}")

    return get_submissions_local()


def get_submissions_local() -> list[dict]:
    """로컬 JSON 파일에서 제출 목록 조회"""
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "training_submissions",
    )
    if not os.path.exists(base):
        return []

    results = []
    for fname in sorted(os.listdir(base)):
        if fname.endswith(".json"):
            with open(os.path.join(base, fname)) as f:
                data = json.load(f)
                data["_source"] = "local"
                data["_file"] = fname
                results.append(data)

    return sorted(results, key=lambda x: x.get("avg_return_pct", 0), reverse=True)


def list_submissions():
    """제출 목록 테이블 출력"""
    subs = get_submissions_from_db()
    if not subs:
        logger.info("제출된 훈련 결과가 없습니다.")
        return

    print(f"\n{'='*90}")
    print(f"  RL 훈련 제출 목록 ({len(subs)}건)")
    print(f"{'='*90}")
    print(f"{'#':>3} | {'Trainer':>15} | {'Algo':>4} | {'Return':>8} | {'Sharpe':>7} | "
          f"{'MDD':>7} | {'Trades':>6} | {'Edge':>4} | {'Status':>10}")
    print("-" * 90)

    for i, s in enumerate(subs):
        status = s.get("status", "submitted")
        edge = "Y" if s.get("edge_cases") else "N"
        print(
            f"{i+1:>3} | "
            f"{s.get('trainer_id', '?'):>15} | "
            f"{s.get('algorithm', '?'):>4} | "
            f"{s.get('avg_return_pct', 0):>7.2f}% | "
            f"{s.get('avg_sharpe', 0):>7.3f} | "
            f"{s.get('avg_mdd', 0):>6.2%} | "
            f"{s.get('avg_trades', 0):>6.1f} | "
            f"{edge:>4} | "
            f"{status:>10}"
        )
    print()


def show_leaderboard():
    """트레이너별 최고 성적 리더보드"""
    subs = get_submissions_from_db()
    if not subs:
        logger.info("데이터 없음")
        return

    # 트레이너별 최고 수익률
    best_by_trainer = {}
    for s in subs:
        tid = s.get("trainer_id", "unknown")
        ret = s.get("avg_return_pct", 0)
        if tid not in best_by_trainer or ret > best_by_trainer[tid]["avg_return_pct"]:
            best_by_trainer[tid] = s

    ranked = sorted(best_by_trainer.values(),
                    key=lambda x: x.get("avg_return_pct", 0), reverse=True)

    print(f"\n{'='*70}")
    print(f"  트레이너 리더보드")
    print(f"{'='*70}")
    print(f"{'순위':>4} | {'Trainer':>15} | {'Best Return':>12} | {'Algo':>4} | {'제출수':>5}")
    print("-" * 70)

    for rank, t in enumerate(ranked, 1):
        tid = t.get("trainer_id", "?")
        count = sum(1 for s in subs if s.get("trainer_id") == tid)
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(
            f"{medal}{rank:>2} | "
            f"{tid:>15} | "
            f"{t.get('avg_return_pct', 0):>11.2f}% | "
            f"{t.get('algorithm', '?'):>4} | "
            f"{count:>5}"
        )
    print()


def promote_model(submission_id: int):
    """특정 제출을 best 모델로 승격"""
    from rl_hybrid.rl.policy import MODEL_DIR

    subs = get_submissions_from_db()
    if submission_id < 1 or submission_id > len(subs):
        logger.error(f"유효하지 않은 번호: {submission_id} (1~{len(subs)})")
        return

    target = subs[submission_id - 1]
    trainer_id = target.get("trainer_id", "?")
    algo = target.get("algorithm", "ppo")
    ret = target.get("avg_return_pct", 0)

    logger.info(f"승격 대상: #{submission_id} {trainer_id} ({algo}) — {ret:.2f}%")

    # 모델 파일 찾기
    submissions_dir = os.path.join(MODEL_DIR, "submissions")
    model_found = False

    if os.path.exists(submissions_dir):
        for dname in sorted(os.listdir(submissions_dir), reverse=True):
            if dname.startswith(trainer_id):
                result_path = os.path.join(submissions_dir, dname, "result.json")
                model_path = os.path.join(submissions_dir, dname, "model.zip")
                if os.path.exists(result_path) and os.path.exists(model_path):
                    with open(result_path) as f:
                        local_result = json.load(f)
                    # Match by model_hash (exact) or avg_return_pct (fallback)
                    target_hash = target.get("model_hash", "")
                    matched = False
                    if target_hash and local_result.get("model_hash") == target_hash:
                        matched = True  # exact hash match
                    elif abs(local_result.get("avg_return_pct", -999) - ret) < 0.01:
                        matched = True  # fallback float comparison
                    if not matched:
                        continue
                    # best에 복사
                    best_dir = os.path.join(MODEL_DIR, "best")
                    os.makedirs(best_dir, exist_ok=True)

                    # 기존 best 백업
                    backup_dir = os.path.join(
                        MODEL_DIR, "best_backup",
                        datetime.now().strftime("%Y%m%d_%H%M%S"),
                    )
                    if os.path.exists(os.path.join(best_dir, "best_model.zip")):
                        os.makedirs(backup_dir, exist_ok=True)
                        for f_name in os.listdir(best_dir):
                            shutil.copy2(
                                os.path.join(best_dir, f_name),
                                os.path.join(backup_dir, f_name),
                            )
                        logger.info(f"기존 best 백업: {backup_dir}")

                    # 새 모델 승격
                    shutil.copy2(model_path, os.path.join(best_dir, "best_model.zip"))
                    model_info = {
                        "algorithm": algo,
                        "observation_dim": 42,
                        "avg_return_pct": ret,
                        "avg_sharpe": target.get("avg_sharpe"),
                        "avg_mdd": target.get("avg_mdd"),
                        "promoted_from": trainer_id,
                        "promoted_at": datetime.now().isoformat(),
                    }
                    with open(os.path.join(best_dir, "model_info.json"), "w") as f:
                        json.dump(model_info, f, indent=2)

                    model_found = True
                    logger.info(f"승격 완료! 새 best: {ret:.2f}% by {trainer_id}")
                    break

    if model_found:
        _update_db_status(target, "promoted")
    else:
        logger.error(
            "모델 파일을 찾을 수 없습니다. "
            "해당 Trainer의 data/rl_models/submissions/ 디렉토리를 확인하세요."
        )
        _update_db_status(target, "rejected")


def _update_db_status(submission: dict, status: str):
    """DB에서 제출 상태 업데이트"""
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    record_id = submission.get("id")

    if not supabase_url or not supabase_key or not record_id:
        return

    try:
        import requests
        requests.patch(
            f"{supabase_url}/rest/v1/rl_training_results?id=eq.{record_id}",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
            },
            json={"status": status, "updated_at": datetime.now().isoformat()},
            timeout=10,
        )
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Admin: 훈련 결과 리뷰")
    parser.add_argument("--list", action="store_true", help="제출 목록")
    parser.add_argument("--leaderboard", action="store_true", help="트레이너 리더보드")
    parser.add_argument("--promote", type=int, help="승격할 제출 번호")

    args = parser.parse_args()

    if args.list:
        list_submissions()
    elif args.leaderboard:
        show_leaderboard()
    elif args.promote:
        promote_model(args.promote)
    else:
        list_submissions()

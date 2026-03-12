"""RL DB 로거 — 모든 RL 훈련/추론/모델 버전을 Supabase에 기록

모든 RL 모듈이 이 모듈을 통해 DB에 기록한다.
REST API(service role key) 사용 — psycopg2 불필요.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("rl.db_logger")

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _post(table: str, data: dict) -> Optional[dict]:
    """Supabase REST API로 단일 레코드 삽입"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase 설정 없음 -- DB 기록 스킵")
        return None

    url = f"{SUPABASE_URL}/rest/v1/{table}"
    # None 값 필터링 + datetime 직렬화
    clean = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, datetime):
            clean[k] = v.isoformat()
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = v
    try:
        r = requests.post(url, headers=_headers(), json=clean, timeout=15)
        if r.status_code in (200, 201):
            result = r.json()
            return result[0] if isinstance(result, list) and result else result
        else:
            logger.error(f"DB insert 실패 [{table}]: {r.status_code} - {r.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"DB insert 예외 [{table}]: {e}")
        return None


def _patch(table: str, match: dict, data: dict) -> bool:
    """Supabase REST API로 레코드 업데이트"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False

    url = f"{SUPABASE_URL}/rest/v1/{table}"
    params = {f"{k}": f"eq.{v}" for k, v in match.items()}
    clean = {k: v.isoformat() if isinstance(v, datetime) else v
             for k, v in data.items() if v is not None}
    try:
        r = requests.patch(url, headers=_headers(), json=clean, params=params, timeout=15)
        return r.status_code in (200, 204)
    except Exception as e:
        logger.error(f"DB update 예외 [{table}]: {e}")
        return False


# ============================================================
# 1. 훈련 사이클 기록
# ============================================================

def log_training_start(
    cycle_type: str,
    algorithm: str,
    module: str,
    training_steps: int = None,
    training_epochs: int = None,
    data_days: int = None,
    data_count: int = None,
    obs_dim: int = 42,
    morl_enabled: bool = False,
    interval: str = "4h",
) -> Optional[str]:
    """훈련 시작 기록 — cycle_id 반환"""
    cycle_id = str(uuid.uuid4())
    result = _post("rl_training_cycles", {
        "id": cycle_id,
        "cycle_type": cycle_type,
        "algorithm": algorithm,
        "module": module,
        "training_steps": training_steps,
        "training_epochs": training_epochs,
        "data_days": data_days,
        "data_count": data_count,
        "obs_dim": obs_dim,
        "morl_enabled": morl_enabled,
        "interval": interval,
        "status": "running",
        "started_at": datetime.now(timezone.utc),
    })
    if result:
        logger.info(f"훈련 사이클 시작 기록: {cycle_id[:8]}... [{algorithm}/{module}]")
        return cycle_id
    return cycle_id  # DB 실패해도 ID는 반환


def log_training_complete(
    cycle_id: str,
    avg_return_pct: float = None,
    avg_sharpe: float = None,
    avg_mdd: float = None,
    avg_trades: float = None,
    policy_loss: float = None,
    value_loss: float = None,
    entropy: float = None,
    direction_accuracy: float = None,
    q_loss: float = None,
    cql_penalty: float = None,
    best_eval_loss: float = None,
    n_sequences: int = None,
    context_length: int = None,
    model_version: str = None,
    model_path: str = None,
    baseline_sharpe: float = None,
    improved: bool = None,
    elapsed_seconds: float = None,
    status: str = "completed",
    error_message: str = None,
):
    """훈련 완료/실패 기록"""
    _patch("rl_training_cycles", {"id": cycle_id}, {
        "avg_return_pct": avg_return_pct,
        "avg_sharpe": avg_sharpe,
        "avg_mdd": avg_mdd,
        "avg_trades": avg_trades,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "direction_accuracy": direction_accuracy,
        "q_loss": q_loss,
        "cql_penalty": cql_penalty,
        "best_eval_loss": best_eval_loss,
        "n_sequences": n_sequences,
        "context_length": context_length,
        "model_version": model_version,
        "model_path": model_path,
        "baseline_sharpe": baseline_sharpe,
        "improved": improved,
        "elapsed_seconds": elapsed_seconds,
        "status": status,
        "error_message": error_message,
        "completed_at": datetime.now(timezone.utc),
    })
    logger.info(f"훈련 사이클 완료 기록: {cycle_id[:8]}... [status={status}]")


# ============================================================
# 2. 추론 예측 기록
# ============================================================

def log_prediction(
    decision_id: str = None,
    cycle_id: str = None,
    ensemble_action: float = None,
    ensemble_direction: str = None,
    num_models: int = None,
    sb3_action: float = None,
    sb3_version: str = None,
    dt_action: float = None,
    dt_version: str = None,
    multi_agent_action: float = None,
    multi_agent_direction: str = None,
    multi_agent_scalp_action: float = None,
    multi_agent_swing_action: float = None,
    offline_action: float = None,
    offline_version: str = None,
    btc_price: float = None,
    rsi_14: float = None,
    fgi: int = None,
    danger_score: float = None,
    opportunity_score: float = None,
) -> Optional[str]:
    """RL 앙상블 추론 결과 기록"""
    pred_id = str(uuid.uuid4())
    result = _post("rl_model_predictions", {
        "id": pred_id,
        "decision_id": decision_id,
        "cycle_id": cycle_id,
        "ensemble_action": ensemble_action,
        "ensemble_direction": ensemble_direction,
        "num_models": num_models,
        "sb3_action": sb3_action,
        "sb3_version": sb3_version,
        "dt_action": dt_action,
        "dt_version": dt_version,
        "multi_agent_action": multi_agent_action,
        "multi_agent_direction": multi_agent_direction,
        "multi_agent_scalp_action": multi_agent_scalp_action,
        "multi_agent_swing_action": multi_agent_swing_action,
        "offline_action": offline_action,
        "offline_version": offline_version,
        "btc_price": btc_price,
        "rsi_14": rsi_14,
        "fgi": fgi,
        "danger_score": danger_score,
        "opportunity_score": opportunity_score,
    })
    if result:
        logger.info(f"추론 기록: {pred_id[:8]}... [dir={ensemble_direction}, models={num_models}]")
    return pred_id


def update_prediction_outcome(
    prediction_id: str,
    price_after_4h: float = None,
    price_after_24h: float = None,
    btc_price_at_prediction: float = None,
):
    """사후 평가 업데이트 (4h/24h 후 가격)"""
    data = {}
    if price_after_4h is not None:
        data["price_after_4h"] = price_after_4h
        if btc_price_at_prediction:
            ret = (price_after_4h - btc_price_at_prediction) / btc_price_at_prediction * 100
            data["return_after_4h"] = round(ret, 4)
    if price_after_24h is not None:
        data["price_after_24h"] = price_after_24h
        if btc_price_at_prediction:
            ret = (price_after_24h - btc_price_at_prediction) / btc_price_at_prediction * 100
            data["return_after_24h"] = round(ret, 4)

    if data:
        _patch("rl_model_predictions", {"id": prediction_id}, data)


# ============================================================
# 3. 모델 버전 기록
# ============================================================

def log_model_version(
    version_id: str,
    algorithm: str,
    model_path: str = None,
    sharpe_ratio: float = None,
    total_return_pct: float = None,
    max_drawdown: float = None,
    eval_episodes: int = None,
    training_steps: int = None,
    training_days: int = None,
    training_config: dict = None,
    is_active: bool = False,
    notes: str = None,
    promoted_from: str = None,
) -> Optional[str]:
    """모델 버전 등록 (registry.json → DB 미러링)"""
    result = _post("rl_model_versions", {
        "version_id": version_id,
        "algorithm": algorithm,
        "model_path": model_path,
        "sharpe_ratio": sharpe_ratio,
        "total_return_pct": total_return_pct,
        "max_drawdown": max_drawdown,
        "eval_episodes": eval_episodes,
        "training_steps": training_steps,
        "training_days": training_days,
        "training_config": training_config,
        "is_active": is_active,
        "notes": notes,
        "promoted_from": promoted_from,
    })
    if result:
        logger.info(f"모델 버전 DB 기록: {version_id} [{algorithm}]")
    return version_id


def update_model_version(version_id: str, **kwargs):
    """모델 버전 정보 업데이트 (is_active, live 성과 등)"""
    _patch("rl_model_versions", {"version_id": version_id}, kwargs)


def deactivate_all_models():
    """모든 모델 비활성화 (새 모델 승격 전)"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    url = f"{SUPABASE_URL}/rest/v1/rl_model_versions"
    try:
        requests.patch(
            url,
            headers=_headers(),
            json={"is_active": False, "updated_at": datetime.now(timezone.utc).isoformat()},
            params={"is_active": "eq.true"},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"모델 비활성화 실패: {e}")


# ============================================================
# 4. 파라미터 튜닝 기록
# ============================================================

def log_parameter_tuning(
    parameter_name: str,
    old_value: float,
    new_value: float,
    change_reason: str = "auto_tuning",
    before_sharpe: float = None,
    after_sharpe: float = None,
    before_return: float = None,
    after_return: float = None,
    approved: bool = None,
    rolled_back: bool = False,
    rollback_reason: str = None,
):
    """Self-Tuning 파라미터 변경 기록"""
    _post("rl_parameter_tuning", {
        "parameter_name": parameter_name,
        "old_value": old_value,
        "new_value": new_value,
        "change_reason": change_reason,
        "before_sharpe": before_sharpe,
        "after_sharpe": after_sharpe,
        "before_return": before_return,
        "after_return": after_return,
        "approved": approved,
        "rolled_back": rolled_back,
        "rollback_reason": rollback_reason,
    })
    logger.info(
        f"파라미터 튜닝 기록: {parameter_name} "
        f"{old_value:.6f} → {new_value:.6f} [{change_reason}]"
    )


# ============================================================
# 5. 쿼리 헬퍼
# ============================================================

def get_recent_training_cycles(
    algorithm: str = None,
    module: str = None,
    limit: int = 20,
) -> list[dict]:
    """최근 훈련 사이클 조회"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    url = f"{SUPABASE_URL}/rest/v1/rl_training_cycles"
    params = {
        "select": "*",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    if algorithm:
        params["algorithm"] = f"eq.{algorithm}"
    if module:
        params["module"] = f"eq.{module}"

    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def get_model_prediction_accuracy(version_id: str = None) -> dict:
    """모델 예측 정확도 조회"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {}

    url = f"{SUPABASE_URL}/rest/v1/rl_model_predictions"
    params = {
        "select": "prediction_quality",
        "prediction_quality": "not.is.null",
    }
    if version_id:
        params["or"] = (
            f"(sb3_version.eq.{version_id},"
            f"dt_version.eq.{version_id},"
            f"offline_version.eq.{version_id})"
        )

    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=10)
        if r.status_code != 200:
            return {}
        results = r.json()
        total = len(results)
        correct = sum(1 for r in results if r["prediction_quality"] == "correct")
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
        }
    except Exception:
        return {}

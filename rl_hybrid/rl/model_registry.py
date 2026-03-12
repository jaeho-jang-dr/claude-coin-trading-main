"""모델 레지스트리 — 버전 관리, 성능 추적, 롤백

훈련된 모델을 버전별로 저장하고, 성능 메트릭을 기록하며,
성능 저하 시 이전 버전으로 롤백한다.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("rl.model_registry")

MODEL_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rl_models",
)


class ModelRegistry:
    """모델 버전 관리 레지스트리"""

    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or MODEL_BASE_DIR
        self.registry_path = os.path.join(self.base_dir, "registry.json")
        os.makedirs(self.base_dir, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "current_version": None,
            "versions": [],
            "rollback_count": 0,
        }

    def _save_registry(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register_model(
        self,
        model_path: str,
        metrics: dict,
        training_config: dict = None,
        notes: str = "",
    ) -> str:
        """새 모델 버전 등록

        Args:
            model_path: 저장된 모델 파일 경로
            metrics: {"sharpe_ratio", "total_return_pct", "max_drawdown", ...}
            training_config: 훈련 설정 (steps, lr, data_days, ...)
            notes: 메모

        Returns:
            버전 ID (예: "v001")
        """
        version_num = len(self.registry["versions"]) + 1
        version_id = f"v{version_num:03d}"

        # 모델을 버전 디렉토리에 복사
        version_dir = os.path.join(self.base_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)

        if os.path.exists(model_path):
            dest = os.path.join(version_dir, os.path.basename(model_path))
            shutil.copy2(model_path, dest)
            model_file = dest
        else:
            model_file = model_path

        version_info = {
            "version_id": version_id,
            "model_path": model_file,
            "metrics": metrics,
            "training_config": training_config or {},
            "notes": notes,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "live_performance": {},
        }

        self.registry["versions"].append(version_info)
        self.registry["current_version"] = version_id
        self._save_registry()

        logger.info(
            f"모델 등록: {version_id}, "
            f"sharpe={metrics.get('sharpe_ratio', 'N/A')}, "
            f"return={metrics.get('total_return_pct', 'N/A')}%"
        )

        # DB 동기화: 기존 활성 모델 비활성화 후 새 버전 등록
        try:
            from rl_hybrid.rl.rl_db_logger import (
                deactivate_all_models as _deactivate_all,
                log_model_version as _log_version,
            )
            _deactivate_all()
            _log_version(
                version_id=version_id,
                algorithm=(training_config or {}).get("algorithm", "unknown"),
                model_path=model_file,
                sharpe_ratio=metrics.get("sharpe_ratio"),
                total_return_pct=metrics.get("total_return_pct"),
                max_drawdown=metrics.get("max_drawdown"),
                eval_episodes=metrics.get("eval_episodes"),
                training_steps=(training_config or {}).get("total_timesteps"),
                training_days=(training_config or {}).get("data_days"),
                training_config=training_config,
                is_active=True,
                notes=notes or None,
            )
        except Exception as e:
            logger.warning(f"모델 등록 DB 동기화 실패 (비치명적): {e}")

        return version_id

    def get_current_version(self) -> Optional[dict]:
        """현재 활성 버전 정보"""
        current = self.registry.get("current_version")
        if not current:
            return None
        return self._get_version(current)

    def get_model_path(self, version_id: str = None) -> Optional[str]:
        """모델 파일 경로 반환"""
        v = self._get_version(version_id or self.registry.get("current_version"))
        return v["model_path"] if v else None

    def _get_version(self, version_id: str) -> Optional[dict]:
        for v in self.registry["versions"]:
            if v["version_id"] == version_id:
                return v
        return None

    def update_live_performance(
        self,
        version_id: str,
        metrics: dict,
    ):
        """라이브 환경에서의 성능 업데이트

        Args:
            version_id: 버전 ID
            metrics: {"trades", "win_rate", "avg_return", "sharpe", ...}
        """
        v = self._get_version(version_id)
        if v:
            v["live_performance"] = {
                **v.get("live_performance", {}),
                **metrics,
                "updated_at": datetime.now().isoformat(),
            }
            self._save_registry()

            # DB 동기화: 라이브 성능 메트릭 업데이트
            try:
                from rl_hybrid.rl.rl_db_logger import update_model_version as _update_ver
                _update_ver(
                    version_id,
                    live_trades=metrics.get("trades"),
                    live_win_rate=metrics.get("win_rate"),
                    live_avg_return=metrics.get("avg_return"),
                    live_sharpe=metrics.get("sharpe"),
                )
            except Exception as e:
                logger.warning(f"라이브 성능 DB 업데이트 실패 (비치명적): {e}")

    def should_rollback(self, version_id: str = None) -> tuple[bool, str]:
        """롤백 필요 여부 판단

        조건:
          - 라이브 승률 < 30% (최소 10거래 이상)
          - 라이브 평균 수익률 < -2%
          - 연속 5회 이상 손실

        Returns:
            (should_rollback: bool, reason: str)
        """
        v = self._get_version(version_id or self.registry.get("current_version"))
        if not v:
            return False, "버전 없음"

        perf = v.get("live_performance", {})
        trades = perf.get("trades", 0)

        if trades < 10:
            return False, f"데이터 부족 ({trades}거래)"

        win_rate = perf.get("win_rate", 0.5)
        avg_return = perf.get("avg_return", 0)
        consecutive_losses = perf.get("consecutive_losses", 0)

        if win_rate < 0.3:
            return True, f"승률 저조: {win_rate:.1%} ({trades}거래)"
        if avg_return < -2.0:
            return True, f"평균 수익률 저조: {avg_return:.2f}%"
        if consecutive_losses >= 5:
            return True, f"연속 {consecutive_losses}회 손실"

        return False, "OK"

    def rollback(self) -> Optional[str]:
        """이전 최고 성능 버전으로 롤백

        Returns:
            롤백된 버전 ID, 불가능하면 None
        """
        versions = self.registry["versions"]
        if len(versions) < 2:
            logger.warning("롤백 불가: 이전 버전 없음")
            return None

        # 최고 샤프 지수 버전 찾기
        best_version = None
        best_sharpe = float("-inf")

        for v in versions[:-1]:  # 현재 버전 제외
            sharpe = v.get("metrics", {}).get("sharpe_ratio", 0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_version = v

        if best_version:
            old_version = self.registry["current_version"]
            self.registry["current_version"] = best_version["version_id"]
            self.registry["rollback_count"] += 1
            self._save_registry()

            logger.info(
                f"모델 롤백: {old_version} → {best_version['version_id']} "
                f"(sharpe={best_sharpe:.3f})"
            )

            # DB 동기화: 이전 버전 비활성화, 롤백 대상 활성화
            try:
                from rl_hybrid.rl.rl_db_logger import update_model_version as _update_ver
                _update_ver(old_version, is_active=False)
                _update_ver(best_version["version_id"], is_active=True)
            except Exception as e:
                logger.warning(f"롤백 DB 동기화 실패 (비치명적): {e}")

            return best_version["version_id"]

        return None

    def list_versions(self) -> list[dict]:
        """모든 버전 목록"""
        return [
            {
                "version_id": v["version_id"],
                "created_at": v["created_at"],
                "sharpe": v.get("metrics", {}).get("sharpe_ratio", "N/A"),
                "return_pct": v.get("metrics", {}).get("total_return_pct", "N/A"),
                "is_current": v["version_id"] == self.registry.get("current_version"),
                "live_trades": v.get("live_performance", {}).get("trades", 0),
            }
            for v in self.registry["versions"]
        ]

    def cleanup_old_versions(self, keep: int = 5):
        """오래된 버전 정리 (디스크 공간 확보)"""
        versions = self.registry["versions"]
        current = self.registry.get("current_version")

        if len(versions) <= keep:
            return

        # 현재 버전 + 최근 keep개 보존
        to_remove = []
        for v in versions[:-keep]:
            if v["version_id"] != current:
                to_remove.append(v)

        for v in to_remove:
            version_dir = os.path.join(self.base_dir, v["version_id"])
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            self.registry["versions"].remove(v)
            logger.info(f"버전 정리: {v['version_id']}")

        self._save_registry()

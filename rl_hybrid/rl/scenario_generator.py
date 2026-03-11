"""엣지 케이스 시나리오 생성기 — 극단 상황 합성 데이터로 RL 모델 강화

실제 시장에서 드물지만 발생하면 치명적인 시나리오를 합성하여
모델이 블랙스완에 대비할 수 있게 훈련한다.

시나리오 목록:
  1. flash_crash       — 30분 내 -15~25% 폭락 후 부분 반등
  2. dead_cat_bounce   — 폭락 후 50% 반등 → 재폭락
  3. parabolic_pump    — 급등 +20~40% → 급락 -30%
  4. whale_manipulation— 고래 펌핑 → 덤핑 반복 (위아래 whipsaw)
  5. sideways_trap     — 3~5일 극도의 횡보(±0.3%) 후 방향성 돌파
  6. cascade_liquidation— 레버리지 청산 연쇄: 급락→반등→재급락→재반등
  7. v_shape_recovery  — 극단 공포(-20%) 후 V자 완전 회복
  8. slow_bleed        — 매일 -1~2%씩 30일 연속 하락 (개미 지옥)
  9. fomo_top          — 극단 탐욕 → 서서히 가속 상승 → 급반전
  10. black_swan       — 1시간 내 -30% + 거래량 10배 → 혼란

사용법:
    from rl_hybrid.rl.scenario_generator import ScenarioGenerator
    gen = ScenarioGenerator(base_price=100_000_000)
    all_candles = gen.generate_all()       # 전체 시나리오 합성
    mixed = gen.mix_with_real(real_candles) # 실제 + 합성 데이터 혼합
"""

import copy
import math
import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


class ScenarioGenerator:
    """극단 시장 시나리오 합성 캔들 생성기"""

    def __init__(self, base_price: float = 100_000_000, seed: int = 42):
        self.base_price = base_price
        self.rng = np.random.RandomState(seed)

    def _make_candle(self, timestamp: str, open_p: float, close_p: float,
                     high_p: float = None, low_p: float = None,
                     volume: float = None) -> dict:
        """단일 캔들 생성"""
        if high_p is None:
            high_p = max(open_p, close_p) * (1 + self.rng.uniform(0, 0.003))
        if low_p is None:
            low_p = min(open_p, close_p) * (1 - self.rng.uniform(0, 0.003))
        if volume is None:
            volume = self.rng.uniform(200, 800)

        return {
            "timestamp": timestamp,
            "open": round(open_p),
            "high": round(max(high_p, open_p, close_p)),
            "low": round(min(low_p, open_p, close_p)),
            "close": round(close_p),
            "volume": round(volume, 4),
        }

    def _timestamps(self, n: int, start: str = "2025-01-01T00:00:00") -> list[str]:
        """n개의 1시간 간격 타임스탬프 생성"""
        base = datetime.fromisoformat(start)
        return [(base + timedelta(hours=i)).isoformat() for i in range(n)]

    def flash_crash(self, severity: float = 0.20) -> list[dict]:
        """시나리오 1: 플래시 크래시
        30분(0.5봉) 내 -15~25% 폭락 후 3~6시간에 걸쳐 50~70% 반등
        """
        candles = []
        ts = self._timestamps(72)
        price = self.base_price

        # 12시간 안정기
        for i in range(12):
            noise = self.rng.uniform(-0.005, 0.005)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        # 폭락 (3봉에 걸쳐)
        crash_bottom = price * (1 - severity)
        crash_steps = [0.5, 0.35, 0.15]  # 1봉에서 50%, 2봉에서 35%, 3봉에서 15%
        for j, pct in enumerate(crash_steps):
            drop = (price - crash_bottom) * pct
            new_price = price - drop
            vol = self.rng.uniform(2000, 5000)  # 거래량 폭증
            candles.append(self._make_candle(ts[12 + j], price, new_price,
                                             volume=vol))
            price = new_price

        # 부분 반등 (20봉에 걸쳐 50~70% 회복)
        recovery_target = crash_bottom + (self.base_price - crash_bottom) * self.rng.uniform(0.5, 0.7)
        recovery_per_step = (recovery_target - price) / 20
        for k in range(20):
            noise = self.rng.uniform(-0.003, 0.005)
            new_price = price + recovery_per_step + price * noise
            vol = self.rng.uniform(800, 2000) * (1 - k / 30)
            candles.append(self._make_candle(ts[15 + k], price, new_price, volume=vol))
            price = new_price

        # 잔여: 불안정 횡보
        for m in range(35, len(ts)):
            noise = self.rng.uniform(-0.008, 0.008)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[m], price, new_price))
            price = new_price

        return candles

    def dead_cat_bounce(self) -> list[dict]:
        """시나리오 2: 데드캣 바운스
        -15% 폭락 → +8% 반등 → 다시 -12% 재폭락
        """
        candles = []
        ts = self._timestamps(96)
        price = self.base_price

        # 안정기 (12시간)
        for i in range(12):
            noise = self.rng.uniform(-0.004, 0.004)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        # 1차 폭락 -15% (4봉)
        for j in range(4):
            drop = self.rng.uniform(0.03, 0.05)
            new_price = price * (1 - drop)
            candles.append(self._make_candle(ts[12 + j], price, new_price,
                                             volume=self.rng.uniform(1500, 3000)))
            price = new_price

        # 바운스 +8% (10봉, "함정" 반등)
        bounce_target = price * 1.08
        bounce_per = (bounce_target - price) / 10
        for k in range(10):
            new_price = price + bounce_per + price * self.rng.uniform(-0.002, 0.004)
            candles.append(self._make_candle(ts[16 + k], price, new_price,
                                             volume=self.rng.uniform(500, 1200)))
            price = new_price

        # 2차 폭락 -12% (5봉)
        for m in range(5):
            drop = self.rng.uniform(0.02, 0.03)
            new_price = price * (1 - drop)
            candles.append(self._make_candle(ts[26 + m], price, new_price,
                                             volume=self.rng.uniform(2000, 4000)))
            price = new_price

        # 잔여 횡보
        for n in range(31, len(ts)):
            noise = self.rng.uniform(-0.006, 0.006)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[n], price, new_price))
            price = new_price

        return candles

    def parabolic_pump(self) -> list[dict]:
        """시나리오 3: 파라볼릭 펌프
        +25~40% 가속 상승 (48시간) → 급락 -30% (6시간)
        """
        candles = []
        ts = self._timestamps(96)
        price = self.base_price

        # 서서히 가속 상승 (48봉)
        for i in range(48):
            accel = (i / 48) ** 2  # 포물선 가속
            gain = 0.002 + accel * 0.015
            noise = self.rng.uniform(-0.002, 0.003)
            new_price = price * (1 + gain + noise)
            vol = self.rng.uniform(300, 600) * (1 + accel * 3)
            candles.append(self._make_candle(ts[i], price, new_price, volume=vol))
            price = new_price

        peak = price

        # 급락 -30% (6봉)
        for j in range(6):
            drop = self.rng.uniform(0.04, 0.07)
            new_price = price * (1 - drop)
            candles.append(self._make_candle(ts[48 + j], price, new_price,
                                             volume=self.rng.uniform(3000, 6000)))
            price = new_price

        # 잔여: 불안정 반등
        for k in range(54, len(ts)):
            noise = self.rng.uniform(-0.008, 0.010)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[k], price, new_price))
            price = new_price

        return candles

    def whale_manipulation(self) -> list[dict]:
        """시나리오 4: 고래 조작 (Whipsaw)
        ±5~8% 급등급락 반복 4회 → 결국 한 방향으로 돌파
        """
        candles = []
        ts = self._timestamps(96)
        price = self.base_price

        for cycle in range(4):
            base_idx = cycle * 12
            direction = 1 if cycle % 2 == 0 else -1
            magnitude = self.rng.uniform(0.05, 0.08)

            # 급등/급락 (3봉)
            for i in range(3):
                move = direction * magnitude / 3
                new_price = price * (1 + move + self.rng.uniform(-0.005, 0.005))
                candles.append(self._make_candle(ts[base_idx + i], price, new_price,
                                                 volume=self.rng.uniform(1500, 3000)))
                price = new_price

            # 되돌림 (3봉)
            for i in range(3):
                move = -direction * magnitude / 4
                new_price = price * (1 + move + self.rng.uniform(-0.003, 0.003))
                candles.append(self._make_candle(ts[base_idx + 3 + i], price, new_price,
                                                 volume=self.rng.uniform(800, 1500)))
                price = new_price

            # 소강 (6봉)
            for i in range(6):
                noise = self.rng.uniform(-0.003, 0.003)
                new_price = price * (1 + noise)
                candles.append(self._make_candle(ts[base_idx + 6 + i], price, new_price))
                price = new_price

        # 돌파 (나머지)
        breakout_dir = self.rng.choice([-1, 1])
        for i in range(48, len(ts)):
            move = breakout_dir * self.rng.uniform(0.003, 0.008)
            new_price = price * (1 + move)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        return candles

    def sideways_trap(self) -> list[dict]:
        """시나리오 5: 횡보 함정
        72시간 ±0.3% 극도의 횡보 → 갑자기 ±10% 돌파
        """
        candles = []
        ts = self._timestamps(96)
        price = self.base_price

        # 72시간 극횡보
        for i in range(72):
            noise = self.rng.uniform(-0.003, 0.003)
            new_price = price * (1 + noise)
            vol = self.rng.uniform(100, 300)  # 거래량 극소
            candles.append(self._make_candle(ts[i], price, new_price, volume=vol))
            price = new_price

        # 돌파 (±10%, 6봉)
        direction = self.rng.choice([-1, 1])
        for i in range(6):
            move = direction * self.rng.uniform(0.012, 0.02)
            new_price = price * (1 + move)
            vol = self.rng.uniform(2000, 4000)
            candles.append(self._make_candle(ts[72 + i], price, new_price, volume=vol))
            price = new_price

        # 후속 추세
        for i in range(78, len(ts)):
            move = direction * self.rng.uniform(0.001, 0.005)
            noise = self.rng.uniform(-0.003, 0.003)
            new_price = price * (1 + move + noise)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        return candles

    def cascade_liquidation(self) -> list[dict]:
        """시나리오 6: 연쇄 청산
        -8% 급락 → +3% 반등 → -10% 재급락 → +5% 반등 → -6% 삼중바닥
        """
        candles = []
        ts = self._timestamps(96)
        price = self.base_price
        idx = 0

        waves = [
            (-0.08, 4, 3000),   # 1차 급락
            (0.03, 6, 1000),    # 반등
            (-0.10, 3, 5000),   # 2차 급락 (더 심함)
            (0.05, 8, 1500),    # 반등
            (-0.06, 3, 4000),   # 3차 급락
            (0.08, 12, 800),    # 최종 회복
        ]

        # 안정기
        for i in range(6):
            noise = self.rng.uniform(-0.003, 0.003)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[idx], price, new_price))
            price = new_price
            idx += 1

        for total_move, steps, base_vol in waves:
            per_step = total_move / steps
            for s in range(steps):
                noise = self.rng.uniform(-0.005, 0.005)
                new_price = price * (1 + per_step + noise)
                vol = base_vol * self.rng.uniform(0.8, 1.5)
                candles.append(self._make_candle(ts[idx], price, new_price, volume=vol))
                price = new_price
                idx += 1
                if idx >= len(ts):
                    break
            if idx >= len(ts):
                break

        # 잔여
        while idx < len(ts):
            noise = self.rng.uniform(-0.005, 0.005)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[idx], price, new_price))
            price = new_price
            idx += 1

        return candles

    def v_shape_recovery(self) -> list[dict]:
        """시나리오 7: V자 회복
        극단 공포 -20% (12시간) → 완전 V자 회복 (24시간)
        """
        candles = []
        ts = self._timestamps(72)
        price = self.base_price

        # 안정기 (6시간)
        for i in range(6):
            noise = self.rng.uniform(-0.003, 0.003)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        peak = price

        # 폭락 -20% (12봉)
        for i in range(12):
            drop = 0.20 / 12 + self.rng.uniform(-0.005, 0.005)
            new_price = price * (1 - drop)
            candles.append(self._make_candle(ts[6 + i], price, new_price,
                                             volume=self.rng.uniform(2000, 5000)))
            price = new_price

        bottom = price

        # V자 회복 (24봉, 원래 가격까지)
        recovery_total = peak - bottom
        for i in range(24):
            progress = (i + 1) / 24
            target = bottom + recovery_total * progress
            noise = self.rng.uniform(-0.003, 0.005)
            new_price = target + target * noise
            vol = self.rng.uniform(1000, 3000) * (1 - progress * 0.5)
            candles.append(self._make_candle(ts[18 + i], price, new_price, volume=vol))
            price = new_price

        # 잔여
        for i in range(42, len(ts)):
            noise = self.rng.uniform(-0.004, 0.006)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        return candles

    def slow_bleed(self) -> list[dict]:
        """시나리오 8: 느린 출혈
        매시간 -0.05~0.15% 씩 30일(720시간) 연속 하락
        중간에 가짜 반등 3~4회 삽입
        """
        candles = []
        ts = self._timestamps(720)
        price = self.base_price
        fake_bounces = sorted(self.rng.choice(range(100, 600), size=4, replace=False))

        for i in range(720):
            if i in fake_bounces:
                # 가짜 반등 (3~5봉)
                bounce_len = self.rng.randint(3, 6)
                for j in range(min(bounce_len, 720 - i)):
                    gain = self.rng.uniform(0.002, 0.006)
                    new_price = price * (1 + gain)
                    candles.append(self._make_candle(ts[min(i + j, 719)], price, new_price,
                                                     volume=self.rng.uniform(500, 1200)))
                    price = new_price
                continue

            if len(candles) >= 720:
                break

            bleed = self.rng.uniform(0.0005, 0.0015)
            noise = self.rng.uniform(-0.001, 0.001)
            new_price = price * (1 - bleed + noise)
            vol = self.rng.uniform(150, 400)
            candles.append(self._make_candle(ts[min(i, 719)], price, new_price, volume=vol))
            price = new_price

        return candles[:720]

    def fomo_top(self) -> list[dict]:
        """시나리오 9: FOMO 천장
        극단 탐욕 구간: 점점 빨라지는 상승 → 급반전 -15%
        """
        candles = []
        ts = self._timestamps(96)
        price = self.base_price

        # 가속 상승 (60봉)
        for i in range(60):
            accel = (i / 60) ** 1.5
            gain = 0.001 + accel * 0.008
            noise = self.rng.uniform(-0.001, 0.003)
            new_price = price * (1 + gain + noise)
            vol = self.rng.uniform(200, 500) * (1 + accel * 5)
            candles.append(self._make_candle(ts[i], price, new_price, volume=vol))
            price = new_price

        # 급반전 -15% (6봉)
        for i in range(6):
            drop = self.rng.uniform(0.02, 0.035)
            new_price = price * (1 - drop)
            candles.append(self._make_candle(ts[60 + i], price, new_price,
                                             volume=self.rng.uniform(4000, 8000)))
            price = new_price

        # 패닉 횡보
        for i in range(66, len(ts)):
            noise = self.rng.uniform(-0.008, 0.008)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        return candles

    def black_swan(self) -> list[dict]:
        """시나리오 10: 블랙 스완
        1시간 내 -30% + 거래량 10배 → 혼란 후 부분 회복
        """
        candles = []
        ts = self._timestamps(72)
        price = self.base_price

        # 안정기
        for i in range(10):
            noise = self.rng.uniform(-0.003, 0.003)
            new_price = price * (1 + noise)
            candles.append(self._make_candle(ts[i], price, new_price))
            price = new_price

        # 블랙 스완 (1봉에 -30%)
        crash_price = price * 0.70
        candles.append(self._make_candle(
            ts[10], price, crash_price,
            high_p=price * 1.02,
            low_p=crash_price * 0.95,
            volume=self.rng.uniform(10000, 20000),
        ))
        price = crash_price

        # 혼란 (극도의 변동, 20봉)
        for i in range(20):
            swing = self.rng.uniform(-0.08, 0.10)
            new_price = price * (1 + swing)
            vol = self.rng.uniform(3000, 8000)
            candles.append(self._make_candle(ts[11 + i], price, new_price,
                                             volume=vol))
            price = new_price

        # 안정화 + 부분 회복
        recovery_target = self.base_price * 0.85
        remaining = len(ts) - 31
        for i in range(remaining):
            progress = i / remaining
            target = price + (recovery_target - price) * 0.05
            noise = self.rng.uniform(-0.005, 0.007)
            new_price = target + target * noise
            candles.append(self._make_candle(ts[31 + i], price, new_price))
            price = new_price

        return candles

    def generate_all(self, variations: int = 3) -> list[dict]:
        """모든 시나리오를 variations만큼 변형하여 생성

        Args:
            variations: 시나리오당 변형 수 (seed 변경)

        Returns:
            모든 시나리오 캔들을 이어붙인 리스트
        """
        scenarios = [
            ("flash_crash", self.flash_crash),
            ("dead_cat_bounce", self.dead_cat_bounce),
            ("parabolic_pump", self.parabolic_pump),
            ("whale_manipulation", self.whale_manipulation),
            ("sideways_trap", self.sideways_trap),
            ("cascade_liquidation", self.cascade_liquidation),
            ("v_shape_recovery", self.v_shape_recovery),
            ("slow_bleed", self.slow_bleed),
            ("fomo_top", self.fomo_top),
            ("black_swan", self.black_swan),
        ]

        all_candles = []
        for var in range(variations):
            self.rng = np.random.RandomState(42 + var * 7)
            for name, gen_func in scenarios:
                candles = gen_func()
                all_candles.extend(candles)

        return all_candles

    def mix_with_real(self, real_candles: list[dict], synthetic_ratio: float = 0.3,
                      variations: int = 2) -> list[dict]:
        """실제 캔들과 합성 시나리오를 혼합

        Args:
            real_candles: 실제 히스토리컬 캔들
            synthetic_ratio: 합성 데이터 비율 (0.3 = 30%)
            variations: 시나리오 변형 수

        Returns:
            혼합된 캔들 리스트 (실제 데이터 뒤에 합성 추가)
        """
        if not real_candles:
            return self.generate_all(variations)

        # base_price를 실제 데이터 기준으로 설정
        self.base_price = real_candles[-1]["close"]

        synthetic = self.generate_all(variations)
        target_synthetic_count = int(len(real_candles) * synthetic_ratio / (1 - synthetic_ratio))

        if len(synthetic) > target_synthetic_count:
            # 랜덤 샘플링 (시나리오 단위로)
            step = max(1, len(synthetic) // target_synthetic_count)
            synthetic = synthetic[::step][:target_synthetic_count]

        return real_candles + synthetic

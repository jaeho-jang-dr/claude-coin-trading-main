#!/usr/bin/env python3
"""
블록체인 고래 추적 (무료 API 전용 — API 키 불필요)

데이터 소스: mempool.space (완전 무료, 키 불필요)

분석:
  - 최근 블록에서 10 BTC 이상 대형 트랜잭션 필터링
  - 대형 트랜잭션 건수/총량으로 고래 활동 수준 판단
  - 멤풀 대기 중인 대형 거래로 향후 변동성 추정

출력: JSON (stdout)

사용법:
    python3 scripts/whale_tracker.py              # 최근 3블록 분석
    python3 scripts/whale_tracker.py --blocks 6   # 최근 6블록 분석
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time

# Windows cp949 인코딩 문제 방지
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── 설정 ──────────────────────────────────────────────

MEMPOOL_API = "https://mempool.space/api"

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RATE_LIMIT_WAIT = 1.0  # mempool.space 무료: 제한 완화적이나 예의상 1초 간격

# 고래 기준: 10 BTC 이상
WHALE_THRESHOLD_BTC = 10.0
MEGA_WHALE_BTC = 100.0  # 100 BTC 이상 = 메가 고래

# 커넥션 재사용을 위한 세션
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """모듈 레벨 requests.Session을 반환한다 (커넥션 풀 재사용)."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Accept": "application/json"})
    return _session


# ── HTTP 헬퍼 ─────────────────────────────────────────


def _get(url: str, label: str = "", parse_json: bool = True):
    """Exponential backoff 포함 GET 요청 (Session 재사용)."""
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json() if parse_json else resp.text.strip()
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(
                    f"[whale_tracker] {label} 429. {wait}초 후 재시도",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(
                f"[whale_tracker] {label} HTTP {resp.status_code}",
                file=sys.stderr,
            )
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(
                f"[whale_tracker] {label} 오류: {e}. {wait}초 후 재시도",
                file=sys.stderr,
            )
            time.sleep(wait)
    return None


# ── 데이터 수집 ───────────────────────────────────────


def _fetch_block_hash(height: int) -> tuple[int, str | None]:
    """블록 해시를 조회한다 (병렬 실행용)."""
    block_hash = _get(
        f"{MEMPOOL_API}/block-height/{height}", f"hash_{height}", parse_json=False
    )
    if block_hash and isinstance(block_hash, str):
        return height, block_hash
    return height, None


def _fetch_block_txs(height: int, block_hash: str) -> dict | None:
    """블록의 트랜잭션을 조회한다 (병렬 실행용)."""
    txs = _get(f"{MEMPOOL_API}/block/{block_hash}/txs/0", f"txs_{height}")
    if txs and isinstance(txs, list):
        return {"height": height, "hash": block_hash, "txs": txs}
    return None


def fetch_recent_blocks(count: int = 3) -> list[dict]:
    """mempool.space에서 최근 N개 블록의 트랜잭션을 조회한다."""
    # 최근 블록 높이
    tip_height = _get(f"{MEMPOOL_API}/blocks/tip/height", "tip_height", parse_json=False)
    if tip_height is None:
        return []
    try:
        tip_height = int(tip_height)
    except (ValueError, TypeError):
        print(f"[whale_tracker] tip_height 파싱 실패: {tip_height}", file=sys.stderr)
        return []

    heights = [tip_height - i for i in range(count)]

    # 1단계: 블록 해시를 병렬 조회
    hash_map: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = {executor.submit(_fetch_block_hash, h): h for h in heights}
        for future in as_completed(futures):
            try:
                height, block_hash = future.result()
                if block_hash:
                    hash_map[height] = block_hash
            except Exception as e:
                h = futures[future]
                print(f"[whale_tracker] hash_{h} 실패: {e}", file=sys.stderr)

    if not hash_map:
        return []

    # 2단계: 블록 트랜잭션을 병렬 조회
    blocks_info = []
    with ThreadPoolExecutor(max_workers=len(hash_map)) as executor:
        futures = {
            executor.submit(_fetch_block_txs, h, bh): h
            for h, bh in hash_map.items()
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    blocks_info.append(result)
            except Exception as e:
                h = futures[future]
                print(f"[whale_tracker] txs_{h} 실패: {e}", file=sys.stderr)

    # 높이 역순 정렬 (최신 블록 우선)
    blocks_info.sort(key=lambda b: b["height"], reverse=True)
    return blocks_info


def fetch_mempool_recent() -> list[dict]:
    """mempool.space에서 최근 멤풀 트랜잭션을 가져온다."""
    data = _get(f"{MEMPOOL_API}/mempool/recent", "mempool_recent")
    return data if isinstance(data, list) else []


# ── 분석 ─────────────────────────────────────────────


def _satoshi_to_btc(sats: int) -> float:
    return sats / 100_000_000


def _estimate_direction(tx: dict) -> str:
    """
    트랜잭션 구조로 방향성을 추정한다.
    - 소수 입력 → 다수 출력 = distribution (매도 압력 가능, 거래소 입금 패턴)
    - 다수 입력 → 소수 출력 = consolidation (매수 후 보관, 거래소 출금 패턴)
    - 소수 입력 → 소수 출력 = direct_transfer (방향 불명)

    거래소 입출금을 100% 구분할 순 없지만, 통계적으로:
    - 1~2 input → 많은 output = 거래소가 출금 처리 (bullish)
    - 많은 input → 1~2 output = 사용자가 거래소에 입금 (bearish)
    """
    vout_count = len(tx.get("vout", []))
    vin_count = len(tx.get("vin", []))

    if vin_count <= 2 and vout_count > 10:
        return "exchange_withdrawal"  # 거래소 출금 (bullish)
    elif vin_count > 5 and vout_count <= 2:
        return "exchange_deposit"     # 거래소 입금 (bearish)
    elif vin_count <= 2 and vout_count <= 2:
        return "direct_transfer"       # 1:1 전송 (neutral)
    elif vout_count <= 5:
        return "consolidation"         # 합치기 (mild bullish)
    else:
        return "distribution"          # 나누기 (mild bearish)


def analyze_block_txs(blocks: list[dict]) -> dict:
    """블록 내 대형 트랜잭션을 분석한다. 방향 추정 포함."""
    whale_txs: list[dict] = []
    total_whale_btc = 0.0
    mega_whale_count = 0
    total_txs_scanned = 0

    # 방향별 집계
    direction_counts = {
        "exchange_withdrawal": 0,
        "exchange_deposit": 0,
        "direct_transfer": 0,
        "consolidation": 0,
        "distribution": 0,
    }
    direction_btc = {
        "exchange_withdrawal": 0.0,
        "exchange_deposit": 0.0,
        "direct_transfer": 0.0,
        "consolidation": 0.0,
        "distribution": 0.0,
    }

    for block in blocks:
        for tx in block.get("txs", []):
            total_txs_scanned += 1

            total_output_sats = sum(
                vout.get("value", 0) for vout in tx.get("vout", [])
            )
            total_btc = _satoshi_to_btc(total_output_sats)

            if total_btc < WHALE_THRESHOLD_BTC:
                continue

            direction = _estimate_direction(tx)
            vout_count = len(tx.get("vout", []))
            vin_count = len(tx.get("vin", []))

            if total_btc >= MEGA_WHALE_BTC:
                mega_whale_count += 1

            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            direction_btc[direction] = direction_btc.get(direction, 0) + total_btc

            whale_txs.append({
                "txid": tx.get("txid", "")[:16] + "...",
                "block_height": block["height"],
                "total_btc": round(total_btc, 4),
                "input_count": vin_count,
                "output_count": vout_count,
                "direction": direction,
                "is_mega_whale": total_btc >= MEGA_WHALE_BTC,
            })
            total_whale_btc += total_btc

    whale_txs.sort(key=lambda x: x["total_btc"], reverse=True)

    # 순 방향 추정
    bullish_btc = direction_btc["exchange_withdrawal"] + direction_btc["consolidation"] * 0.5
    bearish_btc = direction_btc["exchange_deposit"] + direction_btc["distribution"] * 0.5
    net_flow_btc = round(bullish_btc - bearish_btc, 4)

    return {
        "blocks_scanned": len(blocks),
        "txs_scanned": total_txs_scanned,
        "whale_txs_count": len(whale_txs),
        "mega_whale_count": mega_whale_count,
        "total_whale_btc": round(total_whale_btc, 4),
        "direction_counts": direction_counts,
        "direction_btc": {k: round(v, 4) for k, v in direction_btc.items()},
        "net_flow_btc": net_flow_btc,
        "net_flow_signal": (
            "bullish" if net_flow_btc > 5
            else "bearish" if net_flow_btc < -5
            else "neutral"
        ),
        "top_whale_txs": whale_txs[:10],
    }


def analyze_mempool_whales(mempool_txs: list[dict]) -> dict:
    """멤풀 내 대형 미확인 트랜잭션을 분석한다."""
    whale_pending: list[dict] = []
    total_pending_btc = 0.0

    for tx in mempool_txs:
        value_sats = tx.get("value", 0)
        total_btc = _satoshi_to_btc(value_sats)

        if total_btc < WHALE_THRESHOLD_BTC:
            continue

        whale_pending.append({
            "txid": tx.get("txid", "")[:16] + "...",
            "btc": round(total_btc, 4),
            "fee_rate": tx.get("fee", 0),
        })
        total_pending_btc += total_btc

    whale_pending.sort(key=lambda x: x["btc"], reverse=True)

    return {
        "mempool_scanned": len(mempool_txs),
        "whale_pending_count": len(whale_pending),
        "total_pending_whale_btc": round(total_pending_btc, 4),
        "top_pending": whale_pending[:5],
    }


def compute_whale_score(block_analysis: dict, mempool_analysis: dict) -> dict:
    """
    고래 활동 점수: -20 ~ +20
    이제 방향성 포함:
      양수 = 고래 매수 우세 (출금/보관 패턴)
      음수 = 고래 매도 우세 (입금/배포 패턴)
    """
    whale_count = block_analysis.get("whale_txs_count", 0)
    mega_count = block_analysis.get("mega_whale_count", 0)
    pending_count = mempool_analysis.get("whale_pending_count", 0)
    net_flow = block_analysis.get("net_flow_btc", 0)
    net_signal = block_analysis.get("net_flow_signal", "neutral")

    score = 0
    reasons: list[str] = []

    # 1) 활동량 기반 (절대값)
    if mega_count >= 3:
        activity_pts = 10
        reasons.append(f"메가 고래(100+ BTC) {mega_count}건 — 대규모 자금 이동")
    elif mega_count >= 1:
        activity_pts = 5
        reasons.append(f"메가 고래(100+ BTC) {mega_count}건")
    elif whale_count >= 5:
        activity_pts = 3
        reasons.append(f"고래 거래 {whale_count}건 활발")
    elif whale_count == 0:
        activity_pts = -3
        reasons.append("고래 활동 미감지 — 안정적 구간")
    else:
        activity_pts = 0

    # 2) 방향성 기반 (핵심 개선)
    direction_counts = block_analysis.get("direction_counts", {})
    withdrawals = direction_counts.get("exchange_withdrawal", 0)
    deposits = direction_counts.get("exchange_deposit", 0)

    if net_signal == "bullish":
        direction_pts = 8
        reasons.append(f"거래소 출금 우세 (순유출 {net_flow:+.1f} BTC) → 매수 보관 패턴")
    elif net_signal == "bearish":
        direction_pts = -8
        reasons.append(f"거래소 입금 우세 (순유입 {abs(net_flow):.1f} BTC) → 매도 준비 패턴")
    elif withdrawals > deposits:
        direction_pts = 3
        reasons.append(f"출금 {withdrawals}건 > 입금 {deposits}건 (약한 bullish)")
    elif deposits > withdrawals:
        direction_pts = -3
        reasons.append(f"입금 {deposits}건 > 출금 {withdrawals}건 (약한 bearish)")
    else:
        direction_pts = 0

    # 3) 멤풀 대기
    if pending_count >= 3:
        pending_pts = 2
        reasons.append(f"멤풀 대형 대기 {pending_count}건 — 추가 변동 예고")
    else:
        pending_pts = 0

    score = activity_pts + direction_pts + pending_pts
    score = max(-20, min(20, score))

    blocks_scanned = block_analysis.get("blocks_scanned", 0)
    if blocks_scanned >= 3 and whale_count >= 3:
        confidence = "medium"
    elif blocks_scanned >= 1:
        confidence = "low"
    else:
        confidence = "none"

    return {
        "score": score,
        "max_score": 20,
        "confidence": confidence,
        "direction": net_signal,
        "net_flow_btc": net_flow,
        "whale_activity_level": (
            "high" if abs(score) >= 10
            else "moderate" if abs(score) >= 5
            else "low" if whale_count > 0
            else "quiet"
        ),
        "reasons": reasons,
        "note": "고래 활동 + 방향 추정 (거래소 입출금 패턴 기반)",
    }


# ── 메인 ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="블록체인 고래 추적 (무료 API)")
    parser.add_argument(
        "--blocks", type=int, default=3, help="분석할 최근 블록 수 (기본: 3)"
    )
    args = parser.parse_args()

    blocks = fetch_recent_blocks(count=args.blocks)
    mempool_txs = fetch_mempool_recent()

    block_analysis = analyze_block_txs(blocks)
    mempool_analysis = analyze_mempool_whales(mempool_txs)
    whale_score = compute_whale_score(block_analysis, mempool_analysis)

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "source": "mempool.space (free, no API key)",
        "blocks_analyzed": args.blocks,
        "whale_threshold_btc": WHALE_THRESHOLD_BTC,
        "block_analysis": block_analysis,
        "mempool_analysis": mempool_analysis,
        "whale_score": whale_score,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        json.dump({"error": str(e)}, sys.stderr, ensure_ascii=False)
        sys.exit(1)

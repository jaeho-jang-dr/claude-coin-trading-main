"""LLM Context Embedding 기반 RL 상태 확장

Gemini gemini-embedding-001 (3072d) 임베딩을 64d로 압축하여
기존 StateEncoder (42d) 관측 벡터에 결합한다.

확장된 관측 벡터: 42d (원본) + 64d (LLM 컨텍스트) = 106d

사용 흐름:
  1. ContextBuilder가 시장 데이터로 자연어 문맥 생성
  2. LLMStateEncoder가 Gemini 임베딩 → 64d 투사
  3. 기존 관측 벡터에 concat하여 106d 반환
"""

import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("rl.llm_state_encoder")

# PyTorch lazy import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch 미설치 — LLM state encoder 비활성화")

from rl_hybrid.rl.state_encoder import StateEncoder, OBSERVATION_DIM

# 상수
LLM_EMBEDDING_DIM = 3072        # Gemini embedding-001 출력 차원
PROJECTED_DIM = 64              # 압축 후 차원
ENHANCED_OBS_DIM = OBSERVATION_DIM + PROJECTED_DIM  # 42 + 64 = 106
PROJECTION_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rl_models", "llm_projection.pt",
)
CACHE_TTL_SECONDS = 3600  # 1시간


# ────────────────────────────────────────────────────────
#  투사 네트워크 (3072 → 64)
# ────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class ProjectionNetwork(nn.Module):
        """3072d Gemini 임베딩 → 64d 압축 투사 레이어

        구조: Linear(3072, 256) → ReLU → Linear(256, 64) → LayerNorm
        오토인코더 사전 학습으로 가중치 초기화 가능.
        """

        def __init__(
            self,
            input_dim: int = LLM_EMBEDDING_DIM,
            hidden_dim: int = 256,
            output_dim: int = PROJECTED_DIM,
        ):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            self.layer_norm = nn.LayerNorm(output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """임베딩 투사 + LayerNorm 정규화"""
            projected = self.encoder(x)
            return self.layer_norm(projected)

    class ProjectionAutoencoder(nn.Module):
        """투사 레이어 사전 학습용 오토인코더

        인코더(3072→256→64) + 디코더(64→256→3072)
        학습 후 인코더 가중치를 ProjectionNetwork에 전이한다.
        """

        def __init__(
            self,
            input_dim: int = LLM_EMBEDDING_DIM,
            hidden_dim: int = 256,
            latent_dim: int = PROJECTED_DIM,
        ):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """인코딩 + 디코딩

            Returns:
                (reconstructed, latent) 튜플
            """
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent


# ────────────────────────────────────────────────────────
#  ContextBuilder — 구조화 데이터 → 자연어 문맥
# ────────────────────────────────────────────────────────

class ContextBuilder:
    """시장 데이터를 LLM 임베딩용 자연어 문맥으로 변환

    구조화된 시장 데이터(가격, 지표, 감성 등)를 사람이 읽을 수 있는
    간결한 문장으로 조합한다. Gemini embedding API에 전달하기 적합한
    500자 내외의 텍스트를 생성한다.
    """

    def build_context(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        agent_state: dict = None,
    ) -> str:
        """구조화된 데이터 → 자연어 문맥 텍스트

        Args:
            market_data: 시장 데이터 (current_price, indicators 등)
            external_data: 외부 데이터 (sources 딕트)
            portfolio: 포트폴리오 (krw_balance, holdings 등)
            agent_state: 에이전트 상태 (danger_score 등)

        Returns:
            자연어 문맥 문자열
        """
        ag = agent_state or {}
        parts = []

        # 1. 가격 정보
        price = market_data.get("current_price", 0)
        change_24h = market_data.get("change_rate_24h", 0)
        if price:
            price_m = price / 1_000_000
            parts.append(f"BTC at {price_m:.1f}M KRW, 24h change {change_24h * 100:+.1f}%")

        # 2. 기술 지표
        indicators = market_data.get("indicators", {})
        rsi = indicators.get("rsi_14")
        if rsi is not None:
            rsi_label = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
            parts.append(f"RSI {rsi:.0f} ({rsi_label})")

        sma20 = indicators.get("sma_20")
        if sma20 and price:
            sma_dev = (price - sma20) / sma20 * 100
            position = "above" if sma_dev >= 0 else "below"
            parts.append(f"Price {abs(sma_dev):.1f}% {position} SMA20")

        macd = indicators.get("macd", {})
        hist = macd.get("histogram")
        if hist is not None:
            macd_label = "bullish" if hist > 0 else "bearish"
            parts.append(f"MACD histogram {macd_label} ({hist:.0f})")

        bollinger = indicators.get("bollinger", {})
        boll_upper = bollinger.get("upper")
        boll_lower = bollinger.get("lower")
        if boll_upper and boll_lower and price:
            if price >= boll_upper:
                parts.append("Price at Bollinger upper band")
            elif price <= boll_lower:
                parts.append("Price at Bollinger lower band")

        # 3. 외부 데이터
        sources = external_data.get("sources", external_data)

        fgi_data = sources.get("fear_greed", {})
        fgi_current = fgi_data.get("current", {})
        fgi_val = fgi_current.get("value") if isinstance(fgi_current, dict) else fgi_data.get("value")
        if fgi_val is not None:
            fgi_val = int(fgi_val)
            if fgi_val <= 20:
                fgi_label = "extreme fear"
            elif fgi_val <= 40:
                fgi_label = "fear"
            elif fgi_val <= 60:
                fgi_label = "neutral"
            elif fgi_val <= 80:
                fgi_label = "greed"
            else:
                fgi_label = "extreme greed"
            parts.append(f"FGI {fgi_val} ({fgi_label})")

        whale_data = sources.get("whale_tracker", {})
        if isinstance(whale_data, dict):
            ws = whale_data.get("whale_score", {})
            whale_score = ws.get("score", 0) if isinstance(ws, dict) else ws
            if whale_score:
                direction = "accumulation" if whale_score > 0 else "distribution"
                parts.append(f"Whale {direction} detected (score {whale_score:+.0f})")

        binance = sources.get("binance_sentiment", {})
        if isinstance(binance, dict):
            funding_data = binance.get("funding_rate", {})
            funding = funding_data.get("current_rate", 0) if isinstance(funding_data, dict) else 0
            if funding:
                parts.append(f"Binance funding {'negative' if funding < 0 else 'positive'} ({funding:.4f})")

            kimchi_data = binance.get("kimchi_premium", {})
            kimchi = kimchi_data.get("premium_pct", 0) if isinstance(kimchi_data, dict) else 0
            if kimchi:
                label = "premium" if kimchi > 0 else "discount"
                parts.append(f"Kimchi {label} {kimchi:+.1f}%")

            ls_data = binance.get("top_trader_long_short", {})
            ls_ratio = ls_data.get("current_ratio", 1.0) if isinstance(ls_data, dict) else 1.0
            if ls_ratio and abs(ls_ratio - 1.0) > 0.2:
                bias = "long-heavy" if ls_ratio > 1.2 else "short-heavy"
                parts.append(f"L/S ratio {ls_ratio:.2f} ({bias})")

        news_data = sources.get("news_sentiment", {})
        if isinstance(news_data, dict):
            ns = news_data.get("sentiment_score", 0)
            if ns:
                sentiment = "positive" if ns > 20 else "negative" if ns < -20 else "mixed"
                parts.append(f"News sentiment {sentiment} ({ns:+.0f})")

        macro = sources.get("macro", {})
        if isinstance(macro, dict):
            analysis = macro.get("analysis", {})
            macro_score = analysis.get("macro_score", 0) if isinstance(analysis, dict) else 0
            if macro_score:
                macro_label = "favorable" if macro_score > 10 else "unfavorable" if macro_score < -10 else "neutral"
                parts.append(f"Macro environment {macro_label} (score {macro_score:+.0f})")

        # 4. 포트폴리오 상태
        total_eval = portfolio.get("total_eval", 0)
        krw = portfolio.get("krw_balance", 0)
        if total_eval > 0:
            cash_ratio = krw / total_eval * 100
            parts.append(f"Cash ratio {cash_ratio:.0f}%")

        holdings = portfolio.get("holdings", [])
        btc_h = next((h for h in holdings if h.get("currency") == "BTC"), None)
        if btc_h:
            pnl = btc_h.get("profit_loss_pct", 0) or 0
            parts.append(f"BTC position PnL {pnl:+.1f}%")

        # 5. 에이전트 상태
        danger = ag.get("danger_score")
        opportunity = ag.get("opportunity_score")
        if danger is not None:
            parts.append(f"Danger score {danger:.0f}/100")
        if opportunity is not None:
            parts.append(f"Opportunity score {opportunity:.0f}/100")

        cascade = ag.get("cascade_risk")
        if cascade is not None and cascade > 30:
            parts.append(f"Cascade risk elevated at {cascade:.0f}")

        return ", ".join(parts) if parts else "No market data available"


# ────────────────────────────────────────────────────────
#  LLMStateEncoder — StateEncoder + LLM 임베딩
# ────────────────────────────────────────────────────────

class LLMStateEncoder:
    """StateEncoder를 LLM 컨텍스트 임베딩으로 확장한 인코더

    기존 42차원 관측 벡터에 Gemini 임베딩을 64차원으로 압축하여
    결합한 106차원 관측 벡터를 생성한다.

    사용법:
        encoder = LLMStateEncoder()
        obs = encoder.encode(market_data, external_data, portfolio, agent_state)
        # obs.shape == (106,)
    """

    def __init__(
        self,
        projection_path: str = None,
        use_cache: bool = True,
        cache_ttl: int = CACHE_TTL_SECONDS,
    ):
        """
        Args:
            projection_path: 사전 학습된 투사 가중치 경로 (None이면 기본 경로)
            use_cache: 임베딩 캐시 사용 여부
            cache_ttl: 캐시 TTL (초)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch가 필요합니다: pip install torch")

        self.base_encoder = StateEncoder()
        self.context_builder = ContextBuilder()
        self.obs_dim = ENHANCED_OBS_DIM

        # 투사 네트워크
        self.projection = ProjectionNetwork()
        projection_path = projection_path or PROJECTION_SAVE_PATH
        if os.path.exists(projection_path):
            self._load_projection(projection_path)
            logger.info(f"투사 가중치 로드: {projection_path}")
        else:
            logger.warning("투사 가중치 없음 — 랜덤 초기화 사용")
        self.projection.eval()

        # Gemini 클라이언트 (lazy init)
        self._gemini_client = None

        # 캐시
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[np.ndarray, float]] = {}  # key → (embedding, timestamp)

    @property
    def gemini_client(self):
        """Gemini 클라이언트 지연 초기화"""
        if self._gemini_client is None:
            try:
                from rl_hybrid.rag.gemini_client import GeminiClient
                self._gemini_client = GeminiClient()
            except Exception as e:
                logger.error(f"Gemini 클라이언트 초기화 실패: {e}")
                self._gemini_client = None
        return self._gemini_client

    def encode(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        agent_state: dict = None,
    ) -> np.ndarray:
        """원시 데이터 → 106d 확장 관측 벡터

        Returns:
            np.ndarray shape (106,)
        """
        # 기본 42d 벡터
        base_obs = self.base_encoder.encode(
            market_data, external_data, portfolio, agent_state
        )

        # LLM 컨텍스트 64d 벡터
        llm_obs = self._get_llm_embedding(
            market_data, external_data, portfolio, agent_state
        )

        return np.concatenate([base_obs, llm_obs])

    def encode_base_only(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        agent_state: dict = None,
    ) -> np.ndarray:
        """기본 42d 관측 벡터만 반환 (폴백용)"""
        return self.base_encoder.encode(
            market_data, external_data, portfolio, agent_state
        )

    def _get_llm_embedding(
        self,
        market_data: dict,
        external_data: dict,
        portfolio: dict,
        agent_state: dict = None,
    ) -> np.ndarray:
        """LLM 컨텍스트 임베딩 생성 (캐시 + 폴백)

        Returns:
            np.ndarray shape (64,), 투사된 임베딩
        """
        # 컨텍스트 텍스트 생성
        context_text = self.context_builder.build_context(
            market_data, external_data, portfolio, agent_state
        )

        # 캐시 확인
        if self.use_cache:
            cache_key = context_text[:200]  # 앞 200자로 키 생성
            cached = self._cache.get(cache_key)
            if cached is not None:
                embedding, ts = cached
                if time.time() - ts < self.cache_ttl:
                    return embedding

        # Gemini API로 임베딩 생성
        raw_embedding = self._generate_embedding(context_text)

        if raw_embedding is None:
            # 폴백: 제로 벡터
            logger.debug("Gemini 임베딩 실패 → 제로 벡터 폴백")
            projected = np.zeros(PROJECTED_DIM, dtype=np.float32)
        else:
            # 투사 (3072 → 64)
            projected = self._project_embedding(raw_embedding)

        # 캐시 저장
        if self.use_cache:
            self._cache[cache_key] = (projected, time.time())
            # 캐시 크기 제한 (최대 100개)
            if len(self._cache) > 100:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

        return projected

    def _generate_embedding(self, text: str) -> Optional[list[float]]:
        """Gemini API로 3072d 임베딩 생성"""
        client = self.gemini_client
        if client is None:
            return None
        try:
            return client.generate_embedding(text)
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None

    def _project_embedding(self, raw_embedding: list[float]) -> np.ndarray:
        """3072d → 64d 투사

        Args:
            raw_embedding: Gemini 3072d 임베딩

        Returns:
            np.ndarray shape (64,), 정규화된 투사 벡터
        """
        with torch.no_grad():
            tensor = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0)
            projected = self.projection(tensor)
            return projected.squeeze(0).numpy()

    def _load_projection(self, path: str):
        """투사 가중치 로드"""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.projection.load_state_dict(state_dict)

    def clear_cache(self):
        """임베딩 캐시 초기화"""
        self._cache.clear()


# ────────────────────────────────────────────────────────
#  투사 레이어 사전 학습
# ────────────────────────────────────────────────────────

def train_projection(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    similarity_weight: float = 0.1,
    save_path: str = None,
) -> dict:
    """Supabase rag_analysis_vectors 테이블의 과거 임베딩으로
    투사 오토인코더를 사전 학습한다.

    Loss = MSE(reconstruction) + similarity_weight * similarity_preservation

    Args:
        epochs: 학습 에폭
        batch_size: 배치 크기
        learning_rate: 학습률
        similarity_weight: 유사도 보존 손실 가중치
        save_path: 투사 가중치 저장 경로

    Returns:
        {"final_loss", "reconstruction_loss", "similarity_loss", "num_samples"}
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch가 필요합니다: pip install torch")

    save_path = save_path or PROJECTION_SAVE_PATH
    logger.info("투사 레이어 사전 학습 시작")

    # 1. Supabase에서 과거 임베딩 로드
    embeddings = _load_historical_embeddings()
    if len(embeddings) < 10:
        logger.warning(f"과거 임베딩 부족 ({len(embeddings)}건) — 최소 10건 필요")
        return {"final_loss": -1, "num_samples": len(embeddings)}

    logger.info(f"과거 임베딩 로드 완료: {len(embeddings)}건")

    # 2. 텐서 변환
    data = torch.tensor(embeddings, dtype=torch.float32)

    # 3. 오토인코더 학습
    autoencoder = ProjectionAutoencoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    autoencoder.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        # 미니배치 셔플
        indices = torch.randperm(len(data))
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sim = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch = data[batch_idx]

            reconstructed, latent = autoencoder(batch)

            # 재구성 손실
            recon_loss = mse_loss(reconstructed, batch)

            # 유사도 보존 손실: 원본 공간의 코사인 유사도 vs 잠재 공간 유사도
            sim_loss = torch.tensor(0.0)
            if len(batch) >= 2 and similarity_weight > 0:
                # 랜덤 페어 샘플링 (최대 16쌍)
                n_pairs = min(16, len(batch) * (len(batch) - 1) // 2)
                pair_indices = torch.randint(0, len(batch), (n_pairs, 2))

                orig_sim = nn.functional.cosine_similarity(
                    batch[pair_indices[:, 0]], batch[pair_indices[:, 1]], dim=1
                )
                latent_sim = nn.functional.cosine_similarity(
                    latent[pair_indices[:, 0]], latent[pair_indices[:, 1]], dim=1
                )
                sim_loss = mse_loss(latent_sim, orig_sim)

            total_loss = recon_loss + similarity_weight * sim_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sim += sim_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.6f} | "
                f"recon={epoch_recon / max(n_batches, 1):.6f} | "
                f"sim={epoch_sim / max(n_batches, 1):.6f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss

    # 4. 인코더 가중치를 ProjectionNetwork에 전이
    projection = ProjectionNetwork()
    # 오토인코더의 인코더 가중치 복사
    projection.encoder.load_state_dict(autoencoder.encoder.state_dict())

    # 5. 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(projection.state_dict(), save_path)
    logger.info(f"투사 가중치 저장: {save_path}")

    return {
        "final_loss": best_loss,
        "reconstruction_loss": epoch_recon / max(n_batches, 1),
        "similarity_loss": epoch_sim / max(n_batches, 1),
        "num_samples": len(embeddings),
    }


def _load_historical_embeddings() -> list[list[float]]:
    """Supabase rag_analysis_vectors에서 과거 임베딩 로드

    Returns:
        [[float] * 3072, ...] 임베딩 리스트
    """
    try:
        from rl_hybrid.config import config

        url = config.supabase.url
        key = config.supabase.service_role_key
        if not url or not key:
            logger.error("Supabase 설정 누락")
            return []

        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        # 최근 500개 임베딩 로드 (학습에 충분한 양)
        resp = requests.get(
            f"{url}/rest/v1/rag_analysis_vectors",
            headers=headers,
            params={
                "select": "embedding",
                "order": "created_at.desc",
                "limit": 500,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            logger.error(f"임베딩 로드 실패: {resp.status_code}")
            return []

        data = resp.json()
        embeddings = []
        for row in data:
            emb = row.get("embedding")
            if emb and isinstance(emb, list) and len(emb) == LLM_EMBEDDING_DIM:
                embeddings.append(emb)
            elif emb and isinstance(emb, str):
                # pgvector 문자열 형식 파싱
                import json as _json
                try:
                    parsed = _json.loads(emb)
                    if len(parsed) == LLM_EMBEDDING_DIM:
                        embeddings.append(parsed)
                except (ValueError, TypeError):
                    pass

        return embeddings

    except ImportError:
        logger.error("requests 라이브러리 필요")
        return []
    except Exception as e:
        logger.error(f"과거 임베딩 로드 실패: {e}")
        return []


# requests import (top-level은 피하고 함수 내에서만 사용)
try:
    import requests
except ImportError:
    pass


# ────────────────────────────────────────────────────────
#  CLI 진입점
# ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        result = train_projection(epochs=epochs)
        print(f"학습 완료: {result}")
    else:
        print("사용법: python llm_state_encoder.py train [epochs]")
        print(f"관측 차원: {OBSERVATION_DIM} (기본) + {PROJECTED_DIM} (LLM) = {ENHANCED_OBS_DIM}")

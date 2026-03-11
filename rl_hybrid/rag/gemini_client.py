"""Gemini API 클라이언트 — 시장 분석 + 임베딩 생성

Gemini 2.5 Pro로 시장 상황을 분석하고,
gemini-embedding-001로 분석 결과를 3072차원 벡터로 변환한다.
"""

import json
import logging
import time
from typing import Optional

import google.generativeai as genai

from rl_hybrid.config import config
from rl_hybrid.rag.prompts import MARKET_ANALYSIS_PROMPT, EMBEDDING_TEXT_TEMPLATE

logger = logging.getLogger("rag.gemini")


class GeminiClient:
    """Gemini API 래퍼 — 분석 + 임베딩"""

    def __init__(self):
        self.cfg = config.gemini
        if not self.cfg.api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다")

        genai.configure(api_key=self.cfg.api_key)
        self.analysis_model = genai.GenerativeModel(self.cfg.analysis_model)

        # Rate limiting
        self._request_times: list[float] = []

    def _rate_limit(self):
        """RPM 제한 준수"""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        if len(self._request_times) >= self.cfg.rpm_limit:
            sleep_time = 60 - (now - self._request_times[0]) + 1
            logger.info(f"Rate limit 대기: {sleep_time:.1f}초")
            time.sleep(sleep_time)
        self._request_times.append(time.time())

    def analyze_market(
        self,
        market_data: dict,
        external_data: dict,
        rag_context: str = "없음",
    ) -> Optional[dict]:
        """시장 데이터를 Gemini에게 분석 요청 → 구조화된 JSON 반환

        Returns:
            {
                "market_regime": str,
                "confidence": float,
                "key_signals": list[str],
                "risk_assessment": str,
                "recommended_action": str,
                "reasoning": str,
                "time_horizon": str,
                "danger_score_adjustment": int,
                "opportunity_score_adjustment": int,
            }
        """
        self._rate_limit()

        prompt = MARKET_ANALYSIS_PROMPT.format(
            market_data=json.dumps(market_data, ensure_ascii=False, indent=2)[:2000],
            external_data=json.dumps(external_data, ensure_ascii=False, indent=2)[:2000],
            rag_context=rag_context[:1000],
        )

        for attempt in range(self.cfg.max_retries):
            try:
                response = self.analysis_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                    ),
                )

                text = response.text.strip()
                # JSON 블록 추출 (```json ... ``` 감싸기 대응)
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                result = json.loads(text)
                logger.info(
                    f"Gemini 분석 완료: regime={result.get('market_regime')}, "
                    f"action={result.get('recommended_action')}, "
                    f"confidence={result.get('confidence')}"
                )
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON 파싱 실패 (시도 {attempt+1}): {e}")
            except Exception as e:
                logger.error(f"Gemini API 호출 실패 (시도 {attempt+1}): {e}")
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error("Gemini 분석 실패 — 모든 재시도 소진")
        return None

    def generate_embedding(self, text: str) -> Optional[list[float]]:
        """텍스트를 3072차원 임베딩 벡터로 변환

        Args:
            text: 임베딩할 텍스트 (분석 결과 요약)

        Returns:
            3072차원 float 리스트, 실패 시 None
        """
        self._rate_limit()

        try:
            result = genai.embed_content(
                model=f"models/{self.cfg.embedding_model}",
                content=text,
                task_type="retrieval_document",
            )
            embedding = result["embedding"]
            logger.debug(f"임베딩 생성 완료: dim={len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None

    def build_embedding_text(self, analysis: dict, market_snapshot: dict = None) -> str:
        """분석 결과를 임베딩용 텍스트로 변환"""
        return EMBEDDING_TEXT_TEMPLATE.format(
            timestamp=time.strftime("%Y-%m-%d %H:%M"),
            market_regime=analysis.get("market_regime", "unknown"),
            confidence=analysis.get("confidence", 0),
            key_signals=", ".join(analysis.get("key_signals", [])),
            risk_assessment=analysis.get("risk_assessment", "unknown"),
            recommended_action=analysis.get("recommended_action", "hold"),
            reasoning=analysis.get("reasoning", ""),
            btc_price=market_snapshot.get("btc_price", "N/A") if market_snapshot else "N/A",
            rsi=market_snapshot.get("rsi", "N/A") if market_snapshot else "N/A",
            fgi=market_snapshot.get("fgi", "N/A") if market_snapshot else "N/A",
        )

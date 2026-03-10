# RAG (Vector Memory) Implementation Guide for Claude Code

이 문서는 Supabase `pgvector` 기반의 RAG(Retrieval-Augmented Generation) 메모리 시스템을 현재 트레이딩 봇에 구현하기 위한 **Claude Code 전용 지시서**입니다. 이 문서를 읽고 아래의 Task 1부터 3까지를 순서대로 구현해 주세요.

---

## 🎯 목표 (Goal)
현재 AI 봇은 매매 판단을 할 때 `PAST_DECISIONS` 에 가장 최근 10개의 거래 내역을 단순 텍스트(`LIMIT 10`)로 통째로 가져오고 있어 **심각한 토큰 낭비**와 **컨텍스트 미스매치**를 발생시키고 있습니다.
이를 해결하기 위해, 
1) 매매 기록이 생성될 때 **시장 상황(RSI, FGI, BTC 변동성 등)을 문장형 텍스트로 만들어 OpenAI `text-embedding-3-small` 모델 로 벡터화(Embeddings)하여 저장**하고,
2) 새로운 판단을 할 때 현재 시장 상황을 벡터화한 뒤, **Cosine Similarity**를 통해 DB에서 "가장 상황이 비슷했던 과거 매매 3개"만 가져와 요약(`사유 + 성과`)하여 프롬프트에 제공해야 합니다.

---

## 🛠️ 구현해야 할 주요 파트 (Tasks)

### Task 1. Supabase Data Schema 업데이트 (`pgvector` 적용)
새로운 마이그레이션 SQL 스크립트(예: `supabase/migrations/014_decision_embeddings.sql`)를 작성하세요.
1. `CREATE EXTENSION IF NOT EXISTS vector;` 를 통해 pgvector 활성화.
2. `decisions` 테이블에 `state_embedding` 컬럼 추가 (`vector(1536)`).
3. (선택) 조회를 빠르게 하기 위한 HNSW 또는 IVFFlat 인덱스 생성.
4. 유사도 검색을 수행하는 Supabase RPC 함수(PostgreSQL Function) 생성.
   - 이름: `match_similar_decisions`
   - 파라미터: `query_embedding vector(1536)`, `match_limit int`
   - 리턴: `id`, `decision`, `reason`, `action`, `profit_loss`, `similarity` (Cosine distance 기반)

### Task 2. Python 스크립트 수정 (`save_decision.py` 및 `recall_rag.py`)
1. **`save_decision.py`** (또는 매핑되는 파일):
   - `decision` 데이터가 DB에 저장되기 직전(또는 직후 업데이트), 거래 선행 지표들 (RSI, FGI, 현재 가격 변동률, 펀딩비 등)을 조합해 하나의 문장 (예: *"Bitcoin price is 120000000 KRW, RSI is 25.5 (oversold), Fear and Greed Index is 20 (Extreme Fear)..."*)을 만듭니다.
   - OpenAI API(`text-embedding-3-small`)를 호출하여 해당 문장의 임베딩 벡터(`[0.01, -0.02, ...]`)를 받습니다.
   - DB 통신 시 이 배열을 `state_embedding` 필드에 넣어 저장합니다.
2. **`scripts/recall_rag.py` (새 파일 작성)**:
   - CLI 인자로 현재 마켓 상태 JSON 파일(예: `market_data.json`)을 받거나 직접 지표를 수집해 임베딩 문장으로 변환합니다.
   - OpenAI 로 임베딩을 받고, Supabase RPC `match_similar_decisions` 에 질의하여 Top-3 과거 거래를 가져옵니다.
   - 결과를 토큰을 극한으로 절약할 수 있도록 한 줄 요약 포맷으로 압축하여 `stdout` 에 출력합니다.
     - 예: `[유사도 89%] 2026-03-01 매수: RSI 25에서 진입 (손실 -2.5%): 너무 이른 물타기 실패`

### Task 3. 파이프라인 연동 (`run_agents.py` & `run_analysis.sh`)
두 메인 파이프라인에서 과거 기록을 불러오던 방식 수정:
1. 기존의 `LIMIT 10` raw fetch 로직을 삭제합니다.
2. 데이터 수집 Phase 직후, 새로 만든 `scripts/recall_rag.py` 를 호출하여 그 결과물(유사한 과거 거래 요약 3건)을 `PAST_DECISIONS` 변수(또는 JSON 키)에 할당합니다.
3. 이를 프롬프트 템플릿에 "가장 유사했던 과거 경험 3가지" 라는 섹션으로 주입합니다.

---

### 주의사항 (Constraints)
- OpenAI API 키는 기존 `.env` 의 `OPENAI_API_KEY` 를 사용합니다.
- `openai` python 패키지가 필요하다면 `requirements.txt` 에 누락되지 않았는지 확인.
- Supabase RPC 함수 반환값에 `profit_loss` 가 포함되어야 AI가 과거 결정을 "반면교사" 삼을 수 있습니다 (매우 중요).

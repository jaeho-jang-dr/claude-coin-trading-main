# ──────────────────────────────────────────────────────────
# 전체 데이터 수집 + 프롬프트 생성 파이프라인 (Windows용)
#
# 수집한 데이터와 전략을 조합하여 claude -p에 전달할
# 프롬프트를 stdout으로 출력한다.
# ──────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

# .env 로드
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

# 긴급 정지 확인
if ($env:EMERGENCY_STOP -eq "true") {
    Write-Error "EMERGENCY_STOP 활성화됨. 실행 중단."
    exit 1
}

$Python = ".venv\Scripts\python.exe"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SnapshotDir = "data\snapshots\$Timestamp"
New-Item -ItemType Directory -Force -Path $SnapshotDir | Out-Null

Write-Host "[$(Get-Date)] 데이터 수집 시작..." -ForegroundColor Cyan

# 1. 시장 데이터
try { & $Python scripts\collect_market_data.py 2>$null | Out-File "$SnapshotDir\market_data.json" -Encoding UTF8 }
catch { '{"error":"market_data 수집 실패"}' | Out-File "$SnapshotDir\market_data.json" -Encoding UTF8 }

# 2. 공포탐욕지수
try { & $Python scripts\collect_fear_greed.py 2>$null | Out-File "$SnapshotDir\fear_greed.json" -Encoding UTF8 }
catch { '{"error":"fear_greed 수집 실패"}' | Out-File "$SnapshotDir\fear_greed.json" -Encoding UTF8 }

# 3. 뉴스
try { & $Python scripts\collect_news.py 2>$null | Out-File "$SnapshotDir\news.json" -Encoding UTF8 }
catch { '{"error":"news 수집 실패"}' | Out-File "$SnapshotDir\news.json" -Encoding UTF8 }

# 4. 차트 캡처
try { & $Python scripts\capture_chart.py 2>$null | Out-File "$SnapshotDir\chart_paths.json" -Encoding UTF8 }
catch { '{"error":"chart 캡처 실패"}' | Out-File "$SnapshotDir\chart_paths.json" -Encoding UTF8 }

# 5. 포트폴리오
try { & $Python scripts\get_portfolio.py 2>$null | Out-File "$SnapshotDir\portfolio.json" -Encoding UTF8 }
catch { '{"error":"portfolio 조회 실패"}' | Out-File "$SnapshotDir\portfolio.json" -Encoding UTF8 }

Write-Host "[$(Get-Date)] 데이터 수집 완료. 프롬프트 생성 중..." -ForegroundColor Cyan

# 데이터 로드
$Strategy = Get-Content "strategy.md" -Raw -Encoding UTF8
$MarketData = Get-Content "$SnapshotDir\market_data.json" -Raw -Encoding UTF8
$FearGreed = Get-Content "$SnapshotDir\fear_greed.json" -Raw -Encoding UTF8
$News = Get-Content "$SnapshotDir\news.json" -Raw -Encoding UTF8
$Portfolio = Get-Content "$SnapshotDir\portfolio.json" -Raw -Encoding UTF8

# Supabase에서 과거 결정 조회
$PastDecisions = "[]"
if ($env:SUPABASE_URL -and $env:SUPABASE_SERVICE_ROLE_KEY) {
    try {
        $headers = @{
            "apikey"        = $env:SUPABASE_SERVICE_ROLE_KEY
            "Authorization" = "Bearer $($env:SUPABASE_SERVICE_ROLE_KEY)"
        }
        $PastDecisions = Invoke-RestMethod `
            -Uri "$($env:SUPABASE_URL)/rest/v1/decisions?select=*&order=created_at.desc&limit=10" `
            -Headers $headers -Method Get 2>$null | ConvertTo-Json -Depth 10
    }
    catch { $PastDecisions = "[]" }
}

# 미반영 피드백 조회
$Feedback = "[]"
if ($env:SUPABASE_URL -and $env:SUPABASE_SERVICE_ROLE_KEY) {
    try {
        $headers = @{
            "apikey"        = $env:SUPABASE_SERVICE_ROLE_KEY
            "Authorization" = "Bearer $($env:SUPABASE_SERVICE_ROLE_KEY)"
        }
        $Feedback = Invoke-RestMethod `
            -Uri "$($env:SUPABASE_URL)/rest/v1/feedback?select=*&applied=eq.false&order=created_at.desc" `
            -Headers $headers -Method Get 2>$null | ConvertTo-Json -Depth 10
    }
    catch { $Feedback = "[]" }
}

# 로컬 피드백 편향치 (사용자 개입)
$UserBiasState = "{}"
if (Test-Path "data\orchestrator_state.json") {
    try {
        $UserBiasState = Get-Content "data\orchestrator_state.json" -Raw -Encoding UTF8
    }
    catch { $UserBiasState = "{}" }
}

# ETH/BTC 비율 및 도미넌스 데이터 수집
$EthData = "{}"
try {
    $script = @"
import requests, json, statistics
try:
    btc = requests.get('https://api.upbit.com/v1/ticker?markets=KRW-BTC').json()[0]
    eth = requests.get('https://api.upbit.com/v1/ticker?markets=KRW-ETH').json()[0]
    days = requests.get('https://api.upbit.com/v1/candles/days?market=KRW-BTC&count=20').json()
    eth_days = requests.get('https://api.upbit.com/v1/candles/days?market=KRW-ETH&count=20').json()

    current_ratio = eth['trade_price'] / btc['trade_price']
    historical_ratios = [e['trade_price']/b['trade_price'] for e, b in zip(eth_days, days)]
    mean = statistics.mean(historical_ratios)
    stdev = statistics.stdev(historical_ratios) if len(historical_ratios) > 1 else 1

    z_score = (current_ratio - mean) / stdev if stdev > 0 else 0
    vr = (eth['acc_trade_price_24h']/eth['trade_price']) / ((btc['acc_trade_price_24h']/btc['trade_price']) + 1e-9)

    out = {
        'eth_btc_ratio': current_ratio,
        'z_score_20d': z_score,
        'volume_ratio': vr,
        'is_altcoin_season': z_score > 1.5 and vr > 0.08,
        'is_btc_dominance': z_score < -1.0
    }
    print(json.dumps(out))
except Exception as e:
    print('{}')
"@
    $EthData = & $Python -c $script
}
catch { $EthData = "{}" }

# 성과 리뷰 조회
$PerformanceReviews = "[]"
if ($env:SUPABASE_URL -and $env:SUPABASE_SERVICE_ROLE_KEY) {
    try {
        $headers = @{
            "apikey"        = $env:SUPABASE_SERVICE_ROLE_KEY
            "Authorization" = "Bearer $($env:SUPABASE_SERVICE_ROLE_KEY)"
        }
        $PerformanceReviews = Invoke-RestMethod `
            -Uri "$($env:SUPABASE_URL)/rest/v1/performance_reviews?select=*&order=reviewed_at.desc&limit=5" `
            -Headers $headers -Method Get 2>$null | ConvertTo-Json -Depth 10
    }
    catch { $PerformanceReviews = "[]" }
}

$Now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# 프롬프트 출력 (stdout)
@"
당신은 암호화폐 자동매매 AI 트레이더입니다.
아래 데이터를 종합 분석하고, 전략에 따라 매매 결정을 내려주세요.

═══════════════════════════════════════════
[매매 전략]
═══════════════════════════════════════════
$Strategy

═══════════════════════════════════════════
[시장 데이터 - OHLCV, 기술지표]
═══════════════════════════════════════════
$MarketData

═══════════════════════════════════════════
[공포탐욕지수]
═══════════════════════════════════════════
$FearGreed

═══════════════════════════════════════════
[최신 뉴스 (24시간)]
═══════════════════════════════════════════
$News

═══════════════════════════════════════════
[현재 포트폴리오]
═══════════════════════════════════════════
$Portfolio

═══════════════════════════════════════════
[과거 의사결정 (최근 10건)]
═══════════════════════════════════════════
$PastDecisions

═══════════════════════════════════════════
[과거 성과 리뷰 (최근 5건)]
═══════════════════════════════════════════
$PerformanceReviews

═══════════════════════════════════════════
[사용자 피드백 (미반영)]
═══════════════════════════════════════════
$Feedback

═══════════════════════════════════════════
[사용자 수동 피드백 상태 (Bias)]
═══════════════════════════════════════════
$UserBiasState

═══════════════════════════════════════════
[ETH/BTC 및 도미넌스 지표]
═══════════════════════════════════════════
$EthData

═══════════════════════════════════════════
[현재 시각]
═══════════════════════════════════════════
$Now KST

═══════════════════════════════════════════
[지시사항]
═══════════════════════════════════════════

1. 위 모든 데이터를 종합하여 시장 상황을 분석하세요. (ETH/BTC 도미넌스 참고)
2. 전략 문서의 "활성 전략"을 확인하고, 매수/매도/관망 조건과 대조하여 결정하세요.
3. 사용자 수동 피드백 상태(Bias)가 있다면 매수 판단 시 해당 Bias*20점 만큼 강력하게 가중치를 적용하세요.
4. 과거 의사결정과 성과 리뷰(Performance Review)를 보고 실수나 실패 패턴이 있으면 반복하지 마세요.
5. 결정을 내린 후, 아래 순서대로 실행하세요:

   a) 결정이 매수 또는 매도인 경우:
      .venv\Scripts\python.exe scripts\execute_trade.py [bid|ask] KRW-BTC [금액|수량]

   b) 텔레그램 알림 전송:
      .venv\Scripts\python.exe scripts\notify_telegram.py trade "[결정 요약]" "[상세 근거]"

5. 최종 결과를 JSON 형식으로 출력하세요.
"@

#!/bin/bash
# =============================================================================
# VPN 우회하여 Upbit API 호출하는 래퍼
# 사용법: bash scripts/vpn_bypass_api.sh python scripts/get_portfolio.py
# =============================================================================

PROJECT_DIR="/Users/drj00/workspace/blockchain"
VPN_IFACE="utun4"

# VPN 터널 잠깐 내리기
sudo ifconfig "$VPN_IFACE" down 2>/dev/null

# 명령 실행
cd "$PROJECT_DIR"
"$@"
EXIT_CODE=$?

# VPN 터널 복구
sudo ifconfig "$VPN_IFACE" up 2>/dev/null

exit $EXIT_CODE

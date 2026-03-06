#!/bin/bash
# =============================================================================
# Upbit API 트래픽을 VPN 우회하여 실제 IP로 나가게 하는 라우팅 설정
# 사용법: sudo bash scripts/upbit_route.sh [add|remove]
# =============================================================================

GATEWAY="172.30.1.254"
ACTION="${1:-add}"

# Upbit API 서버 IP 목록
UPBIT_IPS=$(dig +short api.upbit.com | grep -E '^[0-9]')

if [ "$ACTION" = "add" ]; then
    echo "Adding Upbit API routes via $GATEWAY (bypass VPN)..."
    for ip in $UPBIT_IPS; do
        sudo route -n add -host "$ip" "$GATEWAY" 2>/dev/null && echo "  + $ip"
    done
    echo "Done. Upbit API will bypass VPN."

elif [ "$ACTION" = "remove" ]; then
    echo "Removing Upbit API routes..."
    for ip in $UPBIT_IPS; do
        sudo route -n delete -host "$ip" 2>/dev/null && echo "  - $ip"
    done
    echo "Done."
else
    echo "Usage: sudo bash $0 [add|remove]"
fi

"""바이낸스 계좌 조회"""
import os, time, hmac, hashlib, requests
from dotenv import dotenv_values

env = dotenv_values(".env")
api_key = env["BINANCE_API_KEY"]
secret_key = env["BINANCE_SECRET_KEY"]

timestamp = str(int(time.time() * 1000))
query = f"timestamp={timestamp}"
signature = hmac.new(secret_key.encode(), query.encode(), hashlib.sha256).hexdigest()

headers = {"X-MBX-APIKEY": api_key}

# Spot account
r = requests.get(
    f"https://api.binance.com/api/v3/account?{query}&signature={signature}",
    headers=headers,
)

if r.status_code == 200:
    data = r.json()
    print(f"=== Binance Spot Account ===")
    print(f"거래 가능: {data.get('canTrade')}")
    print()
    balances = [b for b in data.get("balances", []) if float(b["free"]) > 0 or float(b["locked"]) > 0]
    if balances:
        print(f"{'자산':<8} {'가용':>15} {'잠금':>15}")
        print("-" * 40)
        for b in balances:
            print(f"{b['asset']:<8} {b['free']:>15} {b['locked']:>15}")
    else:
        print("보유 자산 없음")
else:
    print(f"Error: {r.status_code}")
    print(r.text[:500])

# Futures account
print()
query2 = f"timestamp={str(int(time.time() * 1000))}"
sig2 = hmac.new(secret_key.encode(), query2.encode(), hashlib.sha256).hexdigest()
r2 = requests.get(
    f"https://fapi.binance.com/fapi/v2/balance?{query2}&signature={sig2}",
    headers=headers,
)
if r2.status_code == 200:
    print("=== Binance Futures Account ===")
    futures = [f for f in r2.json() if float(f.get("balance", 0)) > 0]
    if futures:
        for f in futures:
            print(f"{f['asset']:<8} balance={f['balance']:>15} unrealizedPnL={f.get('crossUnPnl', '0'):>12}")
    else:
        print("선물 보유 자산 없음")
else:
    print(f"Futures Error: {r2.status_code} {r2.text[:200]}")

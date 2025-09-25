# -*- coding: utf-8 -*-
"""
kis_autotrade_from_levels.py
- '권장 호라이즌 & 트레이드 레벨' CSV를 읽어 KIS Open API로 자동매매 수행
- 절차: 매수 지정가 주문 -> 체결 모니터링 -> 익절/손절 후속 매도
- 모의/실전 자동 분기, 해시키(선택), 간단 재시도/에러 핸들링 포함
"""

import os, time, json, argparse, math
import requests
import pandas as pd

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
# -----------------------------
# 환경설정
# -----------------------------
APP_KEY          = os.getenv("KIS_APP_KEY")
APP_SECRET       = os.getenv("KIS_APP_SECRET")
CANO             = os.getenv("KIS_CANO")            # 계좌 8자리
ACNT_PRDT_CD     = os.getenv("KIS_ACNT_PRDT_CD", "01")
IS_PAPER         = os.getenv("IS_PAPER", "true").lower() == "true"
ORD_QTY_DEFAULT  = int(os.getenv("ORD_QTY_DEFAULT", "1"))

BASE_URL = "https://openapivts.koreainvestment.com:29443" if IS_PAPER else "https://openapi.koreainvestment.com:9443"
TR_BUY   = "VTTC0802U" if IS_PAPER else "TTTC0802U"   # 현금매수
TR_SELL  = "VTTC0801U" if IS_PAPER else "TTTC0801U"   # 현금매도

# 타임아웃/재시도
HTTP_TIMEOUT = 10
POLL_INTERVAL_SEC = 3
POLL_MAX_TRIES = 60  # 체결 확인 최대 3분

# -----------------------------
# 유틸
# -----------------------------
def _assert_env():
    miss = [k for k in ["KIS_APP_KEY","KIS_APP_SECRET","KIS_CANO","KIS_ACNT_PRDT_CD"] if not os.getenv(k)]
    if miss:
        raise RuntimeError(f"환경변수 누락: {miss}. .env에 설정하세요.")

def _headers(token=None, tr_id=None, add_hashkey=None):
    h = {
        "content-type": "application/json",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
    }
    if token:
        h["authorization"] = f"Bearer {token}"
    if tr_id:
        h["tr_id"] = tr_id
    if add_hashkey:
        h["hashkey"] = add_hashkey
    # 개인 고객
    h["custtype"] = "P"
    return h

def get_access_token():
    """
    접근토큰 발급: POST /oauth2/tokenP
    """
    url = f"{BASE_URL}/oauth2/tokenP"
    body = {"grant_type":"client_credentials", "appkey":APP_KEY, "appsecret":APP_SECRET}
    r = requests.post(url, headers={"content-type":"application/json"}, json=body, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j["access_token"]

def get_hashkey(token, body_dict):
    """
    (선택) 해시키 발급: POST /uapi/hashkey
    비필수이나 보안 강화를 위해 옵션 제공
    """
    url = f"{BASE_URL}/uapi/hashkey"
    r = requests.post(url, headers=_headers(token=None), data=json.dumps(body_dict), timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json().get("HASH")

def order_cash(token, code6, qty, price, side="buy", use_hashkey=False):
    """
    현금주문(국내) 지정가:
      side='buy' -> TR_BUY, side='sell' -> TR_SELL
      ORD_DVSN: "00"(지정가) / "01"(시장가)
    """
    tr_id = TR_BUY if side=="buy" else TR_SELL
    path = "/uapi/domestic-stock/v1/trading/order-cash"

    ord_dvsn = "00" if price and price > 0 else "01"
    body = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": code6,                  # 6자리 종목코드
        "ORD_DVSN": ord_dvsn,           # 00=지정가, 01=시장가
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": str(price if price else 0)  # 시장가면 0 또는 생략
    }

    hashkey = get_hashkey(token, body) if use_hashkey else None
    r = requests.post(
        f"{BASE_URL}{path}",
        headers=_headers(token, tr_id, hashkey),
        json=body,
        timeout=HTTP_TIMEOUT
    )
    # 주문요청 자체가 200이 아니면 예외
    r.raise_for_status()
    return r.json()

def inquire_order_fill(token, org_no=None):
    """
    단순 예시: 당일 주문체결 조회로 특정 종목 체결여부 확인하는 로직이 일반적.
    여기선 템플릿만 제공. 실제 운용 시:
    - /uapi/domestic-stock/v1/trading/inquire-... 계열 API로 주문번호/체결수량 조회
    - 또는 웹소켓 체결통보(H0STCNI0 등) 구독 권장
    """
    # TODO: 실제 체결조회 API로 교체. 여기서는 True로 가정하거나 sleep/polling.
    time.sleep(1)
    return True

# -----------------------------
# CSV 로직
# -----------------------------
REQUIRED_COLS = [
    "종목명","종목코드","권장호라이즌",
    "last_close","매수가(entry)","익절가(tp)","손절가(sl)","RR"
]

def load_levels_csv(path):
    df = pd.read_csv(path, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].str.zfill(6)
    return df

# -----------------------------
# 전략 실행
# -----------------------------
def run(csv_path, lots=None, use_hashkey=False):
    _assert_env()
    token = get_access_token()
    print("[INFO] access_token OK")

    df = load_levels_csv(csv_path)
    # 유효 행만 사용(가격/코드 체크)
    df = df.dropna(subset=["종목코드","매수가(entry)","익절가(tp)","손절가(sl)"])
    print(f"[INFO] 대상 종목 수: {len(df)}")

    for _, row in df.iterrows():
        name = row["종목명"]; code = row["종목코드"]
        entry = float(row["매수가(entry)"]); tp = float(row["익절가(tp)"]); sl = float(row["손절가(sl)"])
        qty = int(lots or ORD_QTY_DEFAULT)
        print(f"\n=== {name}({code}) 매수준비: qty={qty}, entry={entry}, tp={tp}, sl={sl} ===")

        # 1) 매수 지정가
        try:
            res_buy = order_cash(token, code, qty, entry, side="buy", use_hashkey=use_hashkey)
            print("[BUY-REQ]", json.dumps(res_buy, ensure_ascii=False))
        except Exception as e:
            print(f"[ERROR] 매수 주문 실패: {e}")
            continue

        # 2) 체결 확인(간단폴링 템플릿)
        filled = False
        for i in range(POLL_MAX_TRIES):
            if inquire_order_fill(token):
                filled = True
                print("[INFO] 매수 체결 확인")
                break
            time.sleep(POLL_INTERVAL_SEC)
        if not filled:
            print("[WARN] 매수 미체결(타임아웃) -> 다음 종목으로")
            continue

        # 3) 익절/손절 후속 매도(지정가). OCO는 API에 직접 없으므로 운용 로직으로 관리.
        #    - 실전운용: 실시간 체결/호가 웹소켓으로 트리거, 또는 예약주문 API 활용.
        try:
            # 익절가 지정가 매도
            res_tp = order_cash(token, code, qty, tp, side="sell", use_hashkey=use_hashkey)
            print("[TP-REQ]", json.dumps(res_tp, ensure_ascii=False))
        except Exception as e:
            print(f"[ERROR] 익절 주문 실패: {e}")

        try:
            # 손절가 지정가 매도(시장가로 운용하려면 price=None, side='sell')
            res_sl = order_cash(token, code, qty, sl, side="sell", use_hashkey=use_hashkey)
            print("[SL-REQ]", json.dumps(res_sl, ensure_ascii=False))
        except Exception as e:
            print(f"[ERROR] 손절 주문 실패: {e}")

    print("\n[DONE] 모든 종목 처리 시도 완료.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="권장 호라이즌 & 레벨 CSV 경로")
    ap.add_argument("--lots", type=int, default=None, help="주문수량(없으면 ORD_QTY_DEFAULT 사용)")
    ap.add_argument("--use-hashkey", action="store_true", help="주문 바디 해시키 사용(선택)")
    args = ap.parse_args()
    run(args.csv, lots=args.lots, use_hashkey=args.use_hashkey)

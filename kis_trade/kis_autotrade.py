# -*- coding: utf-8 -*-
"""
KIS 자동매매 스크립트 (CSV → 주문)
- 신뢰도(confidence) ≥ 임계값만 선별
- 호가(틱) 규칙: KOSPI/KOSDAQ
- 유효기간 의사처리: IOC/FOK (간이 폴링+잔량 취소)
- 브래킷 주문(익절/손절) 자동 연동 (의사 OCO)
- 레이트리밋 대응: 최소 간격 + EGW00201 지수 백오프 재시도
- .env: config.env_dir/.env 우선, 없으면 자동탐색
- 모의/실거래: --paper 플래그로 서버/키/계좌/TR-ID 전환
"""

from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from functools import wraps

import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv

import sys
# Ensure project root is on sys.path so imports like `from DSConfig_3 import cfg` work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DSConfig_3 import config
from makedata.dataset_functions import last_trading_day


# -------------------- 서버/상수 --------------------
REAL_BASE = "https://openapi.koreainvestment.com:9443"
PAPER_BASE = "https://openapivts.koreainvestment.com:29443"

# TR 세트 (실거래 / 모의투자)
TR_REAL = {
    "BUY":     "TTTC0802U",
    "SELL":    "TTTC0801U",
    "CANCEL":  "TTTC0803U",
    "INQUIRE": "TTTC8036R",
    "COND":    "TTTC0902U",
}
TR_VTS = {
    "BUY":     "VTTC0802U",
    "SELL":    "VTTC0801U",
    "CANCEL":  "VTTC0803U",
    "INQUIRE": "VTTC8036R",
    "COND":    "VTTC0902U",
}
def tr_ids(is_paper: bool) -> Dict[str, str]:
    return TR_VTS if is_paper else TR_REAL

# 엔드포인트 (계정 권한/문서에 따라 조정 필요)
ENDPOINTS = {
    "inquire_order": "/uapi/domestic-stock/v1/trading/inquire-ccnl",   # 주문/체결 조회(예시)
    "order_cancel":  "/uapi/domestic-stock/v1/trading/order-rvsecncl", # 정정/취소(예시)
    "conditional_order": "/uapi/domestic-stock/v1/trading/order-conditional",  # 조건주문(예시)
}

# -------------------- 레이트리밋: 쓰로틀 + 재시도 --------------------
THROTTLE_MS_DEFAULT = 400
_last_call_ts = 0.0

def throttle(ms: int = THROTTLE_MS_DEFAULT):
    """거래성 API 호출 사이 최소 간격(ms) 보장"""
    global _last_call_ts
    now = time.monotonic()
    need = (ms / 1000.0) - (now - _last_call_ts)
    if need > 0:
        time.sleep(need)
    _last_call_ts = time.monotonic()

def retry_on_rate_limit(max_tries=3, base_sleep=0.5):
    """EGW00201(초당 거래건수 초과) 시 지수 백오프 재시도"""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            throttle_ms = kwargs.pop("throttle_ms", THROTTLE_MS_DEFAULT)
            sleep = base_sleep
            for i in range(1, max_tries + 1):
                throttle(throttle_ms)
                resp = fn(*args, **kwargs)
                # DRY-RUN 빠른 반환
                if isinstance(resp, dict) and resp.get("note", "").startswith("DRY_RUN"):
                    return resp
                # msg_cd 확인
                msg_cd = None
                if isinstance(resp, dict):
                    r = resp.get("response")
                    if isinstance(r, dict):
                        msg_cd = r.get("msg_cd")
                if msg_cd != "EGW00201":
                    return resp
                # 레이트리밋 → 재시도
                if i == max_tries:
                    return resp
                time.sleep(sleep)
                sleep *= 2
            return resp
        return wrapper
    return deco


# -------------------- 호가(틱) 규칙 --------------------
def default_tick_size(price: float, rule: str = "kospi") -> int:
    p = float(price)
    if rule.lower() == "kosdaq":
        if p < 1_000: return 1
        if p < 5_000: return 5
        if p < 10_000: return 10
        if p < 50_000: return 50
        if p < 100_000: return 100
        return 500
    if p < 1_000: return 1
    if p < 5_000: return 5
    if p < 10_000: return 10
    if p < 50_000: return 50
    if p < 100_000: return 100
    if p < 500_000: return 500
    return 1_000

def round_to_tick(price: float, rule: str = "kospi", direction: str = "nearest") -> int:
    tick = default_tick_size(price, rule)
    p = float(price)
    if direction == "up":
        return int(((p + tick - 1e-9) // tick + 1) * tick) if p % tick != 0 else int(p)
    if direction == "down":
        return int((p // tick) * tick)
    return int(round(p / tick) * tick)


# -------------------- KIS API --------------------
def get_access_token(base: str, appkey: str, appsecret: str) -> str:
    url = f"{base}/oauth2/tokenP"
    payload = {"grant_type": "client_credentials", "appkey": appkey, "appsecret": appsecret}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json().get("access_token", "")

@retry_on_rate_limit(max_tries=3, base_sleep=0.5)
def get_hashkey(base: str, appkey: str, appsecret: str, body: Dict[str, Any], throttle_ms=THROTTLE_MS_DEFAULT) -> str:
    url = f"{base}/uapi/hashkey"
    headers = {"appKey": appkey, "appSecret": appsecret, "content-type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
    r.raise_for_status()
    return r.json().get("HASH", "")

@retry_on_rate_limit(max_tries=3, base_sleep=0.5)
def place_order_cash(
    base: str,
    access_token: str,
    appkey: str,
    appsecret: str,
    TR: Dict[str, str],
    account_no8: str,
    product_cd2: str,
    stock_code6: str,
    side: str,
    qty: int,
    price: int,
    ord_dvsn: str = "00",  # 00=지정가, 01=시장가
    dry_run: bool = True,
    throttle_ms: int = THROTTLE_MS_DEFAULT,
) -> Dict[str, Any]:
    body = {
        "CANO": account_no8,
        "ACNT_PRDT_CD": product_cd2,
        "PDNO": stock_code6,
        "ORD_DVSN": ord_dvsn,
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": "0" if ord_dvsn == "01" else str(int(price)),
    }
    h = get_hashkey(base, appkey, appsecret, body, throttle_ms=throttle_ms)
    url = f"{base}/uapi/domestic-stock/v1/trading/order-cash"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": TR["BUY"] if side.upper() == "BUY" else TR["SELL"],
        "hashkey": h,
    }
    if dry_run:
        return {"url": url, "headers": headers, "body": body, "note": "DRY_RUN - no request sent"}

    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
    try:
        data = r.json()
    except Exception:
        data = {"text": r.text}
    return {"status_code": r.status_code, "response": data}

@retry_on_rate_limit(max_tries=3, base_sleep=0.5)
def inquire_order_status(
    base: str,
    access_token: str,
    appkey: str,
    appsecret: str,
    TR: Dict[str, str],
    account_no8: str,
    product_cd2: str,
    order_no: Optional[str] = None,
    dry_run: bool = True,
    throttle_ms: int = THROTTLE_MS_DEFAULT,
) -> Dict[str, Any]:
    url = f"{base}{ENDPOINTS['inquire_order']}"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": TR["INQUIRE"],
    }
    params = {"CANO": account_no8, "ACNT_PRDT_CD": product_cd2, "ODNO": order_no or ""}
    if dry_run:
        return {"url": url, "headers": headers, "params": params, "note": "DRY_RUN - no request sent"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    try:
        return r.json()
    except Exception:
        return {"text": r.text}

@retry_on_rate_limit(max_tries=3, base_sleep=0.5)
def cancel_unfilled(
    base: str,
    access_token: str,
    appkey: str,
    appsecret: str,
    TR: Dict[str, str],
    account_no8: str,
    product_cd2: str,
    stock_code6: str,
    order_qty: int,
    order_unpr: int,
    orgn_order_no: str,
    dry_run: bool = True,
    throttle_ms: int = THROTTLE_MS_DEFAULT,
) -> Dict[str, Any]:
    url = f"{base}{ENDPOINTS['order_cancel']}"
    body = {
        "CANO": account_no8,
        "ACNT_PRDT_CD": product_cd2,
        "PDNO": stock_code6,
        "ORGN_ODNO": orgn_order_no,
        "ORD_DVSN": "00",
        "ORD_QTY": str(int(order_qty)),
        "ORD_UNPR": str(int(order_unpr)),
        "RVSE_CNCL_DVSN_CD": "02",  # 02=취소
    }
    h = get_hashkey(base, appkey, appsecret, body, throttle_ms=throttle_ms)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": TR["CANCEL"],
        "hashkey": h,
    }
    if dry_run:
        return {"url": url, "headers": headers, "body": body, "note": "DRY_RUN - no request sent"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
    try:
        return r.json()
    except Exception:
        return {"text": r.text}

@retry_on_rate_limit(max_tries=3, base_sleep=0.5)
def place_take_profit_limit(
    base: str,
    access_token: str,
    appkey: str,
    appsecret: str,
    TR: Dict[str, str],
    account_no8: str,
    product_cd2: str,
    stock_code6: str,
    qty: int,
    tp_price: int,
    dry_run: bool = True,
    throttle_ms: int = THROTTLE_MS_DEFAULT,
) -> Dict[str, Any]:
    return place_order_cash(
        base=base,
        access_token=access_token,
        appkey=appkey,
        appsecret=appsecret,
        TR=TR,
        account_no8=account_no8,
        product_cd2=product_cd2,
        stock_code6=stock_code6,
        side="SELL",
        qty=qty,
        price=tp_price,
        ord_dvsn="00",
        dry_run=dry_run,
        throttle_ms=throttle_ms,
    )

@retry_on_rate_limit(max_tries=3, base_sleep=0.5)
def place_stop_loss_condition(
    base: str,
    access_token: str,
    appkey: str,
    appsecret: str,
    TR: Dict[str, str],
    account_no8: str,
    product_cd2: str,
    stock_code6: str,
    qty: int,
    sl_price: int,
    market: bool = True,
    dry_run: bool = True,
    throttle_ms: int = THROTTLE_MS_DEFAULT,
) -> Dict[str, Any]:
    url = f"{base}{ENDPOINTS['conditional_order']}"
    body = {
        "CANO": account_no8,
        "ACNT_PRDT_CD": product_cd2,
        "PDNO": stock_code6,
        "TRGT_UNPR": str(int(sl_price)),        # 발동 트리거
        "ORD_QTY": str(int(qty)),
        "ORD_DVSN": "01" if market else "00",   # 발동 후 시장가/지정가
        "ORD_UNPR": "0" if market else str(int(sl_price)),
        "CNDT_DVSN_CD": "02",                   # 하락 조건(예시)
    }
    h = get_hashkey(base, appkey, appsecret, body, throttle_ms=throttle_ms)
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": appkey,
        "appsecret": appsecret,
        "tr_id": TR["COND"],
        "hashkey": h,
    }
    if dry_run:
        return {"url": url, "headers": headers, "body": body, "note": "DRY_RUN - conditional order not sent"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
    try:
        return r.json()
    except Exception:
        return {"text": r.text}


# -------------------- TIF (IOC/FOK) 시뮬레이션 --------------------
@dataclass
class OrderResult:
    order_no: Optional[str]
    filled_qty: int
    status: Dict[str, Any]

def tif_execute_and_manage(
    base: str,
    access_token: str,
    appkey: str,
    appsecret: str,
    TR: Dict[str, str],
    account_no8: str,
    product_cd2: str,
    code6: str,
    side: str,
    qty: int,
    price: int,
    tif: str = "day",
    fill_wait_sec: float = 2.0,
    dry_run: bool = True,
    throttle_ms: int = THROTTLE_MS_DEFAULT,
) -> OrderResult:
    # 1) 지정가 주문
    resp = place_order_cash(
        base, access_token, appkey, appsecret,
        TR, account_no8, product_cd2, code6, side, qty, price,
        ord_dvsn="00", dry_run=dry_run, throttle_ms=throttle_ms
    )
    order_no = None
    if not dry_run:
        order_no = (resp.get("response") or {}).get("output", {}).get("ODNO")

    # day 또는 DRY-RUN: 즉시 반환
    if dry_run or tif.lower() == "day":
        return OrderResult(order_no=order_no, filled_qty=qty if dry_run else 0, status=resp)

    # 2) 짧게 대기 후 조회
    time.sleep(max(0.2, float(fill_wait_sec)))
    q = inquire_order_status(
        base, access_token, appkey, appsecret, TR,
        account_no8, product_cd2,
        order_no=order_no, dry_run=dry_run, throttle_ms=throttle_ms
    )
    filled = 0  # TODO: 조회 응답에서 실제 체결수량 파싱

    if tif.lower() == "fok":
        if filled < qty:
            cancel_unfilled(
                base, access_token, appkey, appsecret, TR,
                account_no8, product_cd2, code6,
                order_qty=(qty - filled), order_unpr=price,
                orgn_order_no=order_no or "", dry_run=dry_run, throttle_ms=throttle_ms
            )
            return OrderResult(order_no=order_no, filled_qty=0, status=q)
    elif tif.lower() == "ioc":
        if filled < qty:
            cancel_unfilled(
                base, access_token, appkey, appsecret, TR,
                account_no8, product_cd2, code6,
                order_qty=(qty - filled), order_unpr=price,
                orgn_order_no=order_no or "", dry_run=dry_run, throttle_ms=throttle_ms
            )
        return OrderResult(order_no=order_no, filled_qty=filled, status=q)

    return OrderResult(order_no=order_no, filled_qty=filled, status=q)


# -------------------- CSV → 주문 루틴 --------------------
def parse_account_parts(acct: str) -> Tuple[str, str]:
    acct = acct.strip()
    if "-" in acct:
        a, b = acct.split("-", 1)
        return a, b
    return acct[:8], acct[-2:]

def to_int_price(x) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env", help=".env 파일 경로 (기본 ./.env)")
    ap.add_argument("--min-conf", type=float, default=0.5, help="투자가능 최소 신뢰도 임계값(기본 0.5)")
    ap.add_argument("--paper", default=True, action="store_true", help="모의투자 서버 사용")
    ap.add_argument("--dry-run", type=lambda s: str(s).lower() != "false", default=False, help="false일 때만 실주문")
    ap.add_argument("--max-orders", type=int, default=99, help="최대 주문 건수 제한")
    ap.add_argument("--tick-rule", choices=["kospi", "kosdaq"], default="kospi", help="호가 규칙")
    ap.add_argument("--tif", choices=["day", "ioc", "fok"], default="day", help="유효기간(의사처리)")
    ap.add_argument("--fill-wait-sec", type=float, default=2.0, help="IOC/FOK 체결 대기초(간이 폴링)")
    ap.add_argument("--bracket", action="store_true", help="브래킷(익절/손절) 자동 연동")
    ap.add_argument("--no-tp", action="store_true", help="익절 비활성화")
    ap.add_argument("--no-sl", action="store_true", help="손절 비활성화")
    ap.add_argument("--sl-market", action="store_true", help="손절 발동 시 시장가")
    ap.add_argument("--tp-offset-ticks", type=int, default=0, help="TP 추가 틱 오프셋(하향)")
    ap.add_argument("--sl-offset-ticks", type=int, default=0, help="SL 추가 틱 오프셋(상향)")
    ap.add_argument("--throttle-ms", type=int, default=THROTTLE_MS_DEFAULT, help="거래성 API 최소 간격(ms)")
    args = ap.parse_args()

    # 1) .env 로드 (config.env_dir/.env 우선 → 없으면 자동탐색)
    env_path = Path(config.env_dir) / ".env"
    if not env_path.exists():
        auto = find_dotenv(usecwd=True)
        if auto:
            env_path = Path(auto)
            print(f"[INFO] 자동탐색된 .env 사용: {env_path}")
        else:
            print(f"[WARN] .env를 찾지 못했습니다: {env_path} (cwd={Path.cwd()})")
    load_dotenv(dotenv_path=str(env_path), override=True, verbose=True)

    # 2) 환경변수 (모의투자 시 *_VTS 지원)
    is_paper = bool(args.paper)
    appkey    = os.getenv("KIS_APPKEY", "")
    appsecret = os.getenv("KIS_APPSECRET", "")
    account   = os.getenv("KIS_ACCOUNT", "")  # 예: 12345678-01
    if is_paper:
        appkey    = os.getenv("KIS_APPKEY_VTS",    appkey)
        appsecret = os.getenv("KIS_APPSECRET_VTS", appsecret)
        account   = os.getenv("KIS_ACCOUNT_VTS",   account)

    if not (appkey and appsecret and account):
        raise RuntimeError(f".env/환경변수에 인증값이 없습니다. (env: {env_path})")

    # 계좌번호(필요 시 parse_account_parts 사용)
    #account_n80, prodcut_cd2 = parse_account_parts(account)  # 유효성 검사
    account_no8 = account
    product_cd2 = "01"

    # 3) 서버/ TR 세트
    base = PAPER_BASE if is_paper else REAL_BASE
    TR = tr_ids(is_paper)

    # 4) 토큰
    access_token = get_access_token(base, appkey, appsecret)

    # 5) CSV 로드/필터/정렬
    config.end_date = last_trading_day()
    input_csv = Path(config.report_dir) / f"Report_{config.end_date}" / f"Trading_price_{config.end_date}.csv"
    df = pd.read_csv(input_csv, dtype={"종목코드": str})
    if "confidence" not in df.columns:
        raise RuntimeError("CSV에 'confidence' 컬럼이 없습니다.")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df[df["confidence"] >= float(args.min_conf)].copy()
    if df.empty:
        print("[INFO] 조건을 만족하는 종목이 없습니다.")
        return
    df = df.sort_values(["confidence", "RR"], ascending=[False, False], kind="mergesort")

    # 6) 주문 루프
    placed = 0
    for _, r in df.iterrows():
        if placed >= args.max_orders:
            break

        code6 = (r.get("종목코드") or "").zfill(6)
        name = r.get("종목명")
        side = str(r.get("side") or "").upper()
        qty = int(float(r.get("ord_qty") or 0))
        if qty <= 0:
            print(f"[SKIP] 수량 0: {name}({code6})")
            continue

        entry = r.get("매수가(entry)")
        if pd.isna(entry) or float(entry) <= 0:
            entry = r.get("last_close")

        px = round_to_tick(float(entry), args.tick_rule, "up" if side == "BUY" else "down")

        print(f"[ORDER] {side} {name}({code6}) x {qty} @ {px}  (tif={args.tif}, dry_run={args.dry_run}, throttle_ms={args.throttle_ms}, paper={is_paper})")

        # 6-1) 진입 주문 (TIF 의사처리)
        res = tif_execute_and_manage(
            base=base,
            access_token=access_token,
            appkey=appkey,
            appsecret=appsecret,
            TR=TR,
            account_no8=account_no8,
            product_cd2=product_cd2,
            code6=code6,
            side=side,
            qty=qty,
            price=px,
            tif=args.tif,
            fill_wait_sec=args.fill_wait_sec,
            dry_run=bool(args.dry_run),
            throttle_ms=int(args.throttle_ms),
        )
        print(json.dumps({"tif_result": res.status}, ensure_ascii=False, indent=2))
        filled_qty = res.filled_qty if not args.dry_run else qty  # DRY-RUN: 전량 체결 가정

        # 6-2) 브래킷(익절/손절)
        if args.bracket and filled_qty > 0:
            tp = r.get("익절가(tp)")
            sl = r.get("손절가(sl)")
            tp_valid = (not pd.isna(tp)) and (float(tp) > 0)
            sl_valid = (not pd.isna(sl)) and (float(sl) > 0)

            if not args.no_tp and tp_valid:
                tp_rounded = round_to_tick(float(tp), args.tick_rule, "down")
                if args.tp_offset_ticks:
                    tp_rounded -= default_tick_size(tp_rounded, args.tick_rule) * int(args.tp_offset_ticks)
                    tp_rounded = max(tp_rounded, 1)
                print(f"[TP]  SELL {name}({code6}) x {filled_qty} @ {tp_rounded}")
                tp_resp = place_take_profit_limit(
                    base, access_token, appkey, appsecret, TR,
                    account_no8, product_cd2,
                    stock_code6=code6, qty=int(filled_qty), tp_price=int(tp_rounded),
                    dry_run=bool(args.dry_run), throttle_ms=int(args.throttle_ms),
                )
                print(json.dumps({"tp_order": tp_resp}, ensure_ascii=False, indent=2))

            if not args.no_sl and sl_valid:
                sl_rounded = round_to_tick(float(sl), args.tick_rule, "up")
                if args.sl_offset_ticks:
                    sl_rounded += default_tick_size(sl_rounded, args.tick_rule) * int(args.sl_offset_ticks)
                print(f"[SL]  STOP {'MKT' if args.sl_market else 'LMT'} {name}({code6}) x {filled_qty} @ {sl_rounded}")
                sl_resp = place_stop_loss_condition(
                    base, access_token, appkey, appsecret, TR,
                    account_no8, product_cd2,
                    stock_code6=code6, qty=int(filled_qty), sl_price=int(sl_rounded),
                    market=bool(args.sl_market), dry_run=bool(args.dry_run), throttle_ms=int(args.throttle_ms),
                )
                print(json.dumps({"sl_order": sl_resp}, ensure_ascii=False, indent=2))

        placed += 1

    print(f"[DONE] 대상 {len(df)}건 중 {placed}건 처리")


if __name__ == "__main__":
    main()

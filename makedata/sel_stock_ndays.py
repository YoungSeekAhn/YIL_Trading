# -*- coding: utf-8 -*-
"""
score_universe.py
- KOSPI200 종목을 대상으로 조건별 점수 산정
- 기술적(6), 거래량(2), 수급(5) = 총 13개 항목 → 최대 100점
- 최근 영업일 5일 점수 합산(Score_1w) 기준 상위 종목 출력/저장
"""

import warnings
warnings.filterwarnings("ignore")

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock as krx

from tqdm import tqdm
from pathlib import Path
import os
from pandas.tseries.offsets import BDay

from DSConfig_3 import DSConfig

cfg = DSConfig

# ===================== 설정 =====================
TOP_N_MARCAP = 100           # 시총 상위 N (현재는 KOSPI200로 대체)
PICK_TOP = 10                # 최종 선별 수
INDEX_TICKER = "KS11"        # KOSPI 지수 (FinanceDataReader)
LOOKBACK_DAYS = 200          # 지표 계산기간(영업일 여유 포함)
WEEK_BDAYS = 5               # '직전 1주' = 최근 5 영업일

# 점수 테이블(요청안)
WEIGHTS = {
    # --- 기술적 ---
    "rsi30": 10,                # (1) RSI < 30
    "momentum_pos": 10,         # (2) MOM>0
    "macd_cross": 10,           # (3) MACD 골든크로스
    "ema5_over_ema20": 5,       # (4) 5일선이 20일선 돌파
    "ema20_over_ema60": 10,     # (5) 20일선이 60일선 돌파
    "rs_plus": 10,              # (6) 20일 상대강도(종목-KOSPI) +

    # --- 거래량 ---
    "vol_120": 5,               # (7) 3일평균 / 20일평균 ≥ 1.2
    "vol_150": 10,              # (8) 3일평균 / 20일평균 ≥ 1.5

    # --- 수급 ---
    "frg_own_1m_up": 10,        # (9) 외국인 지분율 1개월 증가
    "frg_3_pos": 5,             # (10) 외국인 3일 연속 순매수
    "frg_5_pos": 5,             # (11) 외국인 5일 연속 순매수
    "ins_3_pos": 5,             # (12) 기관 3일 연속 순매수
    "ins_5_pos": 5,             # (13) 기관 5일 연속 순매수
}

# 거래량 점수 합산 정책: 120%와 150%가 동시 충족 시
VOLUME_SCORE_MODE = "max"  # "max"=상한만 채택, "sum"=둘 다 더함
# =================================================


# ===================== 유틸/헬퍼 =====================
def add_indicators(px: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터프레임(px)에 기술적 지표 추가:
      - RSI(7), MOM10/5, EMA5/20/60, MACD(12,26,9), VOL_3MA, VOL_MA20
    """
    out = px.copy()
    rename_map = {c: c.capitalize() for c in out.columns}
    out = out.rename(columns=rename_map)
    for need in ["Open", "High", "Low", "Close", "Volume"]:
        if need not in out.columns:
            raise ValueError(f"'{need}' column missing in price DataFrame")

    # RSI(7)
    delta = out["Close"].diff()
    up = delta.clip(lower=0.0).rolling(7).mean()
    down = (-delta.clip(upper=0.0)).rolling(7).mean()
    rs = up / (down + 1e-12)
    out["RSI"] = 100 - (100 / (1 + rs))

    # Momentum(10/5)
    out["MOM10"] = out["Close"] - out["Close"].shift(10)
    out["MOM5"] = out["Close"] - out["Close"].shift(5)

    # EMA
    out["EMA5"] = out["Close"].ewm(span=5, adjust=False).mean()
    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA60"] = out["Close"].ewm(span=60, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIG"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # 거래량 이동평균
    out["VOL_3MA"] = out["Volume"].rolling(3).mean()
    out["VOL_MA20"] = out["Volume"].rolling(20).mean()

    return out


def fetch_investor_netbuy_df(ticker: str, start_yyyymmdd: str, end_yyyymmdd: str) -> pd.DataFrame:
    """
    투자자별 순매수(금액) – pykrx
    반환: index=datetime, columns=['FRG','INS']  (원 단위, +면 순매수)
    """
    if isinstance(ticker, str) and ticker.isdigit() and len(ticker) < 6:
        ticker = ticker.zfill(6)

    df = krx.get_market_trading_value_by_date(start_yyyymmdd, end_yyyymmdd, ticker)
    if df is None or df.empty:
        return pd.DataFrame(columns=["FRG", "INS"])

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    col_map = {
        "FRG": ["외국인", "외국인합계"],
        "INS": ["기관합계", "기관"]
    }

    def pick_col(frame: pd.DataFrame, candidates: List[str]) -> pd.Series:
        for c in candidates:
            if c in frame.columns:
                s = pd.to_numeric(frame[c], errors="coerce")
                s.name = None
                return s
        return pd.Series(index=frame.index, dtype=float)

    s_frg = pick_col(df, col_map["FRG"])
    s_ins = pick_col(df, col_map["INS"])
    out = pd.concat([s_frg.rename("FRG"), s_ins.rename("INS")], axis=1)

    if out["FRG"].isna().all() and out["INS"].isna().all():
        return pd.DataFrame(columns=["FRG", "INS"])

    return out


def foreign_ownership_ratio(date_yyyymmdd: str, market="KOSPI") -> pd.DataFrame:
    """
    외국인 보유비중/한도소진율 등 지표 – 버전마다 컬럼명이 다를 수 있어 추론.
    반환: ['티커','지분율'] 로 정규화
    """
    try:
        df = krx.get_exhaustion_rates_of_foreign_investment_by_ticker(date_yyyymmdd, market=market)
        if "티커" not in df.columns:
            df = df.reset_index().rename(columns={"index": "티커"})
        cand = [c for c in df.columns if ("보유" in c) or ("소진" in c) or ("외국" in c)]
        col = cand[0] if cand else df.columns[-1]
        return df[["티커", col]].rename(columns={col: "지분율"})
    except Exception:
        return pd.DataFrame(columns=["티커", "지분율"])


# ===================== 점수 계산 =====================
@dataclass
class ScoreBreakdown:
    # 기술적
    rsi30: int = 0
    momentum_pos: int = 0
    macd_cross: int = 0
    ema5_over_ema20: int = 0
    ema20_over_ema60: int = 0
    rs_plus: int = 0
    # 거래량
    vol_120: int = 0
    vol_150: int = 0
    # 수급
    frg_own_1m_up: int = 0
    frg_3_pos: int = 0
    frg_5_pos: int = 0
    ins_3_pos: int = 0
    ins_5_pos: int = 0

    def total(self) -> int:
        vol_part = max(self.vol_120, self.vol_150) if VOLUME_SCORE_MODE == "max" else (self.vol_120 + self.vol_150)
        base_sum = (self.rsi30 + self.momentum_pos + self.macd_cross +
                    self.ema5_over_ema20 + self.ema20_over_ema60 + self.rs_plus +
                    self.frg_own_1m_up + self.frg_3_pos + self.frg_5_pos +
                    self.ins_3_pos + self.ins_5_pos)
        return base_sum + (vol_part)


def score_one(
    ticker: str,
    px: pd.DataFrame,               # OHLCV + 인디케이터 (index=Datetime)
    inv: pd.DataFrame,              # 투자자 순매수 금액 ['FRG','INS'] (index=Datetime)
    kospi_close: pd.Series,         # KOSPI 종가 (index=Datetime)
    frg_now: float = np.nan,        # 외국인 지분율(해당일)
    frg_1m: float = np.nan          # 외국인 지분율(해당일의 1개월 전 근사)
) -> Tuple[int, ScoreBreakdown]:
    """
    요청하신 13개 조건으로 점수 산정 (해당일 기준)
    """
    bd = ScoreBreakdown()
    if px is None or px.empty or len(px) < 65:
        return 0, bd

    last = px.index.max()

    # (1) RSI < 30
    try:
        if px.loc[last, "RSI"] < 30:
            bd.rsi30 = WEIGHTS["rsi30"]
    except Exception:
        pass

    # (2) Momentum > 0 (5일)
    try:
        if px.loc[last, "MOM5"] > 0:
            bd.momentum_pos = WEIGHTS["momentum_pos"]
    except Exception:
        pass

    # (3) MACD 골든크로스
    try:
        macd, sig = px["MACD"], px["MACD_SIG"]
        if (macd.iloc[-1] > sig.iloc[-1]) and (macd.iloc[-2] <= sig.iloc[-2]):
            bd.macd_cross = WEIGHTS["macd_cross"]
    except Exception:
        pass

    # (4) 5일선 > 20일선으로 오늘 돌파
    try:
        now = px["EMA5"].iloc[-1] > px["EMA20"].iloc[-1]
        prev = px["EMA5"].iloc[-2] <= px["EMA20"].iloc[-2]
        if now and prev:
            bd.ema5_over_ema20 = WEIGHTS["ema5_over_ema20"]
    except Exception:
        pass

    # (5) 20일선 > 60일선으로 오늘 돌파
    try:
        now = px["EMA20"].iloc[-1] > px["EMA60"].iloc[-1]
        prev = px["EMA20"].iloc[-2] <= px["EMA60"].iloc[-2]
        if now and prev:
            bd.ema20_over_ema60 = WEIGHTS["ema20_over_ema60"]
    except Exception:
        pass

    # (6) 최근 20일 KOSPI 대비 상대 강도 +
    try:
        common = px.index.intersection(kospi_close.index)
        if len(common) >= 21:
            r_stock = px.loc[common, "Close"].pct_change(20).iloc[-1]
            r_kospi = kospi_close.loc[common].pct_change(20).iloc[-1]
            if (r_stock - r_kospi) > 0:
                bd.rs_plus = WEIGHTS["rs_plus"]
    except Exception:
        pass

    # (7)(8) 거래량: 3일평균 / 20일평균
    try:
        ratio = px.loc[last, "VOL_3MA"] / (px.loc[last, "VOL_MA20"] + 1e-12)
        if ratio >= 1.2:
            bd.vol_120 = WEIGHTS["vol_120"]
        if ratio >= 1.5:
            bd.vol_150 = WEIGHTS["vol_150"]
            if VOLUME_SCORE_MODE == "max":
                bd.vol_120 = 0
    except Exception:
        pass

    # (9) 외국인 지분율 1개월 증가
    try:
        if np.isfinite(frg_now) and np.isfinite(frg_1m) and (frg_now - frg_1m) > 0:
            bd.frg_own_1m_up = WEIGHTS["frg_own_1m_up"]
    except Exception:
        pass

    # (10)(11) 외국인 연속 순매수
    try:
        if inv is not None and not inv.empty:
            inv = inv.sort_index()
            if len(inv) >= 3 and (inv["FRG"].iloc[-3:] > 0).all():
                bd.frg_3_pos = WEIGHTS["frg_3_pos"]
            if len(inv) >= 5 and (inv["FRG"].iloc[-5:] > 0).all():
                bd.frg_5_pos = WEIGHTS["frg_5_pos"]
    except Exception:
        pass

    # (12)(13) 기관 연속 순매수
    try:
        if inv is not None and not inv.empty:
            if len(inv) >= 3 and (inv["INS"].iloc[-3:] > 0).all():
                bd.ins_3_pos = WEIGHTS["ins_3_pos"]
            if len(inv) >= 5 and (inv["INS"].iloc[-5:] > 0).all():
                bd.ins_5_pos = WEIGHTS["ins_5_pos"]
    except Exception:
        pass

    return bd.total(), bd


# ========= 주간(최근 5영업일) 합산 유틸 =========
def pick_foreign_own_for_date(fr_df: pd.DataFrame, the_date: pd.Timestamp):
    """
    fr_df: get_exhaustion_rates_of_foreign_investment_by_date 결과
           index=datetime, '한도소진율' 컬럼 가정
    the_date 기준의 최신값(frg_now)과 1개월 전 근사(frg_1m; 22BD 전)를 반환
    """
    if fr_df is None or fr_df.empty or "한도소진율" not in fr_df.columns:
        return np.nan, np.nan

    fr_df = fr_df.sort_index()
    # the_date 이전(포함) 관측치 중 최신
    frg_now = fr_df.loc[:the_date, "한도소진율"].tail(1)
    frg_now = float(frg_now.iloc[0]) if len(frg_now) else np.nan

    d_1m = the_date - BDay(22)
    frg_1m = fr_df.loc[:d_1m, "한도소진율"].tail(1)
    frg_1m = float(frg_1m.iloc[0]) if len(frg_1m) else np.nan
    return frg_now, frg_1m


def compute_weekly_score(
    ticker: str,
    px: pd.DataFrame,
    inv: pd.DataFrame,
    kospi_close: pd.Series,
    fr_df: pd.DataFrame,
) -> Tuple[float, Dict, List[Dict]]:
    """
    최근 5영업일 각각에 대해 score_one(...)을 호출 → 합산
    반환:
      - week_sum: 주간 총점(float)
      - last_bd_dict: 가장 최근 일자의 ScoreBreakdown dict
      - daily_breakdown: [{date, score, **breakdown}, ...] 최근 5영업일 리스트
    """
    if px is None or px.empty:
        return 0.0, {}, []

    px = px.sort_index()
    last_dates = list(px.index[-WEEK_BDAYS:])  # 최근 5BD

    week_sum = 0.0
    last_bd_dict: Dict = {}
    daily_breakdown: List[Dict] = []

    for d in last_dates:
        # 일자 d 까지로 슬라이스
        px_d = px.loc[:d]
        inv_d = None
        if inv is not None and not inv.empty:
            inv_d = inv.copy()
            if not isinstance(inv_d.index, pd.DatetimeIndex):
                inv_d.index = pd.to_datetime(inv_d.index, errors="coerce")
            inv_d = inv_d.sort_index().loc[:d]

        kospi_d = kospi_close.loc[:d]

        frg_now_d, frg_1m_d = pick_foreign_own_for_date(fr_df, d)

        total_d, bd_d = score_one(
            ticker=ticker,
            px=px_d,
            inv=inv_d,
            kospi_close=kospi_d,
            frg_now=frg_now_d,
            frg_1m=frg_1m_d
        )

        week_sum += float(total_d)

        bd_dict = getattr(bd_d, "__dict__", {})
        daily_breakdown.append({"date": d, "score": float(total_d), **bd_dict})

        if d == last_dates[-1]:
            last_bd_dict = bd_dict

    return week_sum, last_bd_dict, daily_breakdown


# ===================== 메인 파이프라인 =====================
def sel_stock(cfg):
    # 날짜 설정
    today = dt.date.today()
    start = today - dt.timedelta(days=LOOKBACK_DAYS)
    s_fdr = start.strftime("%Y-%m-%d")
    e_fdr = today.strftime("%Y-%m-%d")
    s_krx = start.strftime("%Y%m%d")
    e_krx = today.strftime("%Y%m%d")
    d1m  = (today - dt.timedelta(days=30)).strftime("%Y%m%d")

    print(f"[기간] {s_fdr} ~ {e_fdr}")

    # 2) KOSPI 지수 (상대강도용)
    idx_df = fdr.DataReader(INDEX_TICKER, s_fdr, e_fdr)
    if "Close" not in idx_df.columns:
        idx_df.rename(columns={c: c.capitalize() for c in idx_df.columns}, inplace=True)
    kospi_close = idx_df["Close"]
    kospi_close.index = pd.to_datetime(kospi_close.index)

    # (선택) 외국인 지분율 맵 - 본 로직은 fr_df로 일자별 추출하므로 사용 안 함
    fr_now_df = foreign_ownership_ratio(e_krx, market="KOSPI")
    fr_1m_df  = foreign_ownership_ratio(d1m,  market="KOSPI")
    fr_now_map = dict(zip(fr_now_df.get("티커", []), fr_now_df.get("지분율", [])))
    fr_1m_map  = dict(zip(fr_1m_df.get("티커", []),  fr_1m_df.get("지분율", [])))

    # KOSPI200 구성종목 티커(6자리) 리스트
    tickers = krx.get_index_portfolio_deposit_file("1028")

    # 종목명 붙이기
    uni = pd.DataFrame({"Code": tickers})
    uni["Name"] = uni["Code"].apply(krx.get_market_ticker_name)

    rows: List[Dict] = []

    for _, r in tqdm(uni.iterrows(), total=len(uni)):
        ticker = r["Code"]          # 예: '005930'
        name   = r.get("Name", "")

        try:
            # 4) 가격/거래량 (KRX OHLCV) + 인디케이터
            px = krx.get_market_ohlcv_by_date(s_krx, e_krx, ticker)
            if px is None or px.empty:
                continue

            px = px.rename(columns={
                "시가": "Open",
                "고가": "High",
                "저가": "Low",
                "종가": "Close",
                "거래량": "Volume"
            })[["Open","High","Low","Close","Volume"]]
            px.index = pd.to_datetime(px.index)
            px = add_indicators(px)

            # 5) 투자자 수급
            inv = fetch_investor_netbuy_df(ticker, s_krx, e_krx)

            # 6) 외국인 지분율(일자별) - 소진율 시계열
            fr_df = krx.get_exhaustion_rates_of_foreign_investment_by_date(s_krx, e_krx, ticker)
            if fr_df is not None and not fr_df.empty:
                fr_df.index = pd.to_datetime(fr_df.index)
                fr_df = fr_df.sort_index()

            # 7) 최근 5영업일 점수 합산
            week_sum, last_bd_dict, daily_breakdown = compute_weekly_score(
                ticker=ticker,
                px=px,
                inv=inv,
                kospi_close=kospi_close,
                fr_df=fr_df if (fr_df is not None and not fr_df.empty) else pd.DataFrame()
            )

            rows.append({
                "Name": name,
                "Code": ticker,
                "Score_1w": week_sum,     # ★ 주간 총점
                **last_bd_dict            # 마지막 일자 브레이크다운(원하면 보존)
                # 필요 시 daily_breakdown를 별도 파일로 저장 가능
            })

        except Exception:
            # 휴장/데이터 누락/형식 차이 등은 스킵
            continue

    if not rows:
        print("※ 결과 없음: 네트워크/거래일/pykrx 반환 형식 등을 확인하세요.")
        return

    out = pd.DataFrame(rows).sort_values(["Score_1w"], ascending=[False])

    # 출력
    print("\n=== 직전 1주(5영업일) 총점 상위 종목 ===")
    print(out.head(PICK_TOP)[["Name", "Code", "Score_1w"]])

    print("\n=== (참고) 마지막 일자 점수 브레이크다운(상위) ===")
    cols = ["Name","Code","Score_1w"] + list(WEIGHTS.keys())
    cols = [c for c in cols if c in out.columns]
    print(out.head(PICK_TOP)[cols])

    # 저장
    get_dir = Path(cfg.selout_dir)
    get_dir.mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(get_dir, f"scored_{e_krx}.csv")
    out.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[저장] {filepath}")


if __name__ == "__main__":
    sel_stock()

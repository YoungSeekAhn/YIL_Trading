# -*- coding: utf-8 -*-
"""
batch_select_horizon_and_levels_v4.py
- v4 개선판 (고정 비중 수량계산, 변동성/틱/신뢰도/사이드/캘린더 보강)
- <predict_dir>/<end_date>/*.csv 각 파일 단위 처리 → 종목 단위 집계 후 최종 CSV 출력

핵심 변경점(v3 → v4):
  1) buf 산출식 강화: max(MAE, |Bias|, ATR*k) (기본 k=0.6, ATR기간=14)
  2) 틱/호가단위 반영: entry/tp/sl을 틱 단위로 반올림 (거래소별 틱 룰 함수 주입 가능)
  3) 신뢰도 스코어 스무딩: 로지스틱(sigmoid) 기반으로 급경사 완화
  4) 사이드 결정식 고도화: Δp/vol 기반 z-score + DirAcc 보정
  5) 휴일/영업일 훅: 간단한 휴일 리스트를 받아 valid_until 조정 (없으면 주말만 보정)
  6) 품질 플래그 및 설명 컬럼 확장: 선택 사유/페널티 내역을 결과에 일부 노출
  7) 포지션 사이징: per_share_risk 하한을 틱단위로, 수수료/슬리피지 보정 훅 추가

CLI 예:
  python batch_select_horizon_and_levels_v4.py --predict-dir ./predict --end-date 20250925 \
      --report-dir ./reports --out Trading_price_20250925.csv --equity 50000000 --risk-pct 0.005 --allow-short
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Callable
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None
import math
import logging

# ====== 설정 ======
HORIZONS = ["h1", "h2", "h3"]
TARGETS = ["high", "low", "close"]
EPS = 1e-12

# 권장 기준(보수적)
MAPE_MAX = 2.5
DIRACC_MIN = 55.0
BIAS_MAE_RATIO_MAX = 0.5
COUNT_MIN = 40

# 점수 가중치(호라이즌 선택)
W_MAPE = 0.40
W_DA = 0.30
W_BIAS = 0.20
W_MAE = 0.10

# 트레이딩 파라미터
RISK_REWARD = 1.8
ENTRY_PULLBACK = 0.25  # entry = last_close - 0.25*buf_close

# 신뢰도 가중치 (0~1 스케일)
WC_MAPE = 0.40
WC_DA = 0.30
WC_BIAS = 0.20
WC_CNT = 0.10

# 신뢰도 스무딩 파라미터
SIG_K_DA = 0.15   # DirAcc 로지스틱 기울기
SIG_K_M = 1.25   # MAPE 로지스틱 기울기

# ATR 설정
USE_ATR_IN_BUF = True
ATR_PERIOD = 14
ATR_WEIGHT = 0.6   # buf = max(MAE, |Bias|, ATR*0.6)

# 수수료/슬리피지 (양방향 합산 예상치)
SLIPPAGE = 0.0
FEES_PCT = 0.0   # 0.0005 = 5bps

# 마감 시각(현물)
DEFAULT_CLOSE_HHMM = (15, 20)  # 15:20 KST

# 로깅
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ====== 유틸 ======
def _safe_num(s): return pd.to_numeric(s, errors="coerce")

def _calc_atr(df: pd.DataFrame, period: int=14) -> float:
    cols = {c for c in df.columns}
    if not {"true_high", "true_low", "true_close"}.issubset(cols):
        return np.nan
    dd = df.sort_values("date").copy()
    h, l, c = map(_safe_num, (dd["true_high"], dd["true_low"], dd["true_close"]))
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if tr.dropna().empty:
        return np.nan
    atr = tr.rolling(period, min_periods=max(2, period//2)).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else np.nan

# 틱/호가단위: 거래소별 사용자 정의 가능
# 기본 틱: 가격 구간 무시하고 고정 tick_size 반환
TickSizeFn = Callable[[float, str], float]

def default_tick_size(price: float, code: str) -> float:
    # 예시: 코스피/코스닥 일반 단일 틱 10원 (실전은 가격대별 구간 적용 권장)
    return 10.0


def round_to_tick(x: float, tick: float, mode: str="nearest") -> float:
    if not (tick and tick > 0):
        return float(x)
    q = x / tick
    if mode == "down":
        return math.floor(q) * tick
    elif mode == "up":
        return math.ceil(q) * tick
    else:
        return round(q) * tick


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = _safe_num(y_true); y_pred = _safe_num(y_pred)
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"count": 0, "RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan, "Bias": np.nan}
    e = (y_pred[mask] - y_true[mask]).values
    yt = y_true[mask].values
    rmse = float(np.sqrt(np.mean(e**2)))
    mae = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e / np.clip(yt, EPS, None))) * 100.0)
    bias = float(np.mean(e))
    return {"count": int(mask.sum()), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias}


def _diracc_close(df: pd.DataFrame, h: str) -> Dict[str, float]:
    if "true_close" not in df.columns: return {"DirAcc_close(%)": np.nan, "Dir_count": 0}
    pcol = f"pred_{h}_close"
    if pcol not in df.columns: return {"DirAcc_close(%)": np.nan, "Dir_count": 0}
    df = df.sort_values("date")
    y_true = _safe_num(df["true_close"])
    y_pred = _safe_num(df[pcol])
    prev_true = y_true.shift(1)
    mask = y_true.notna() & y_pred.notna() & prev_true.notna()
    if mask.sum() == 0: return {"DirAcc_close(%)": np.nan, "Dir_count": 0}
    true_dir = np.sign(y_true[mask] - prev_true[mask]).values
    pred_dir = np.sign(y_pred[mask] - prev_true[mask]).values
    acc = float((true_dir == pred_dir).mean() * 100.0)
    return {"DirAcc_close(%)": acc, "Dir_count": int(mask.sum())}


def _rank_norm(vals, ascending=True):
    s = pd.Series(vals, dtype=float)
    if s.dropna().empty: return [0.5]*len(s)
    r = s.rank(ascending=ascending, method="average")
    norm = (r - 1) / max(len(s)-1, 1)
    norm[s.isna()] = 0.5
    return norm.tolist()


def _score_hblock(row, h):
    return {
        "MAPE": row.get(f"{h}_close_MAPE(%)", np.nan),
        "DA": row.get(f"{h}_close_DirAcc(%)", np.nan),
        "BiasAbs": abs(row.get(f"{h}_close_Bias", np.nan)) if pd.notna(row.get(f"{h}_close_Bias", np.nan)) else np.nan,
        "MAE": row.get(f"{h}_close_MAE", np.nan),
        "Count": row.get(f"{h}_close_count", np.nan)
    }


def pick_horizon(row) -> Tuple[str, Dict, Dict]:
    hs = HORIZONS
    blocks = {h: _score_hblock(row, h) for h in hs}
    mape_n = _rank_norm([blocks[h]["MAPE"] for h in hs], ascending=True)
    da_n = _rank_norm([blocks[h]["DA"] for h in hs], ascending=False)
    bias_n = _rank_norm([blocks[h]["BiasAbs"] for h in hs], ascending=True)
    mae_n = _rank_norm([blocks[h]["MAE"] for h in hs], ascending=True)
    base = {h: (W_MAPE * mape_n[i] + W_DA * da_n[i] + W_BIAS * bias_n[i] + W_MAE * mae_n[i]) for i, h in enumerate(hs)}
    pen = {h: 0.0 for h in hs}
    flags = {h: [] for h in hs}
    for h in hs:
        b = blocks[h]
        if pd.notna(b["MAPE"]) and b["MAPE"] > MAPE_MAX:
            pen[h] += 0.50
            flags[h].append("MAPE>MAX")
        if pd.notna(b["DA"]) and b["DA"] < DIRACC_MIN:
            pen[h] += 0.25
            flags[h].append("DA<MIN")
        if pd.notna(b["MAE"]) and pd.notna(b["BiasAbs"]) and (b["BiasAbs"] > BIAS_MAE_RATIO_MAX * b["MAE"]):
            pen[h] += 0.20
            flags[h].append("|Bias|>r*MAE")
        if pd.notna(b["Count"]) and b["Count"] < COUNT_MIN:
            pen[h] += 0.25
            flags[h].append("COUNT<MIN")
    total = {h: base[h] + pen[h] for h in hs}
    best = sorted(hs, key=lambda x: (
        total[x],
        blocks[x]["MAPE"] if pd.notna(blocks[x]["MAPE"]) else np.inf,
        -(blocks[x]["DA"] if pd.notna(blocks[x]["DA"]) else -np.inf)
    ))[0]
    explain = {h: {"base": round(base[h], 4), "penalty": round(pen[h], 4), "total": round(total[h], 4)} for h in hs}
    return best, explain, flags


def _buf(mae, bias, atr):
    vals = [mae, abs(bias), atr * ATR_WEIGHT if pd.notna(atr) else np.nan]
    vals = [v for v in vals if pd.notna(v)]
    return max(vals) if vals else np.nan

def compute_levels(row, h_sel, last_close, last_pred: Dict[str, float],
                   atr_val: float,
                   code: str,
                   tick_fn: TickSizeFn = default_tick_size):
    mae = row.get(f"{h_sel}_close_MAE", np.nan)
    bias= row.get(f"{h_sel}_close_Bias", np.nan)
    mae_h= row.get(f"{h_sel}_high_MAE", np.nan); bias_h = row.get(f"{h_sel}_high_Bias", np.nan)
    mae_l= row.get(f"{h_sel}_low_MAE",  np.nan); bias_l = row.get(f"{h_sel}_low_Bias",  np.nan)

    atr = atr_val if USE_ATR_IN_BUF else np.nan
    buf_close = _buf(mae, bias, atr)
    # high/low 전용 버퍼
    atr_h = atr_val if USE_ATR_IN_BUF else np.nan
    atr_l = atr_val if USE_ATR_IN_BUF else np.nan
    buf_high  = _buf(mae_h, bias_h, atr_h) if (pd.notna(mae_h) or pd.notna(bias_h)) else buf_close
    buf_low   = _buf(mae_l, bias_l, atr_l) if (pd.notna(mae_l) or pd.notna(bias_l)) else buf_close

    if pd.isna(buf_close):
        buf_close = max(1.0, mae if pd.notna(mae) else 1.0)

    # 기본 레벨
    entry = float(last_close) - ENTRY_PULLBACK * buf_close
    risk  = buf_close
    tp    = float(last_close) + RISK_REWARD * risk
    sl    = float(last_close) - risk

    # 예측 상한/하한으로 보수 조정
    p_close = last_pred.get("close", np.nan)
    p_high  = last_pred.get("high",  np.nan)
    p_low   = last_pred.get("low",   np.nan)
    if not pd.isna(p_high) or not pd.isna(p_close):
        candidates = [v for v in [p_high, p_close] if pd.notna(v)]
        best_up = np.min(candidates) if candidates else np.nan
        if pd.notna(best_up):
            tp = min(tp, best_up - 0.5*buf_high)
    if not pd.isna(p_low):
        sl = max(sl, p_low + 0.5*buf_low)

    # 수수료/슬리피지 보정
    tp = tp * (1 - FEES_PCT) - SLIPPAGE
    sl = sl * (1 + FEES_PCT) + SLIPPAGE

    # 틱 반올림
    t_entry = tick_fn(entry, code)
    t_tp    = tick_fn(tp, code)
    t_sl    = tick_fn(sl, code)
    entry = round_to_tick(entry, t_entry, mode="down")     # 매수는 보수적으로 낮게
    tp    = round_to_tick(tp,    t_tp,    mode="down")     # 익절은 약간 낮게
    sl    = round_to_tick(sl,    t_sl,    mode="up")       # 손절은 약간 높게(더 타이트)

    rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else np.nan
    return {
        "매수가(entry)": round(entry, 2),
        "익절가(tp)"   : round(tp, 2),
        "손절가(sl)"   : round(sl, 2),
        "RR"          : round(rr, 2),
        "buf_close"   : round(buf_close, 3),
        "buf_high"    : round(float(buf_high), 3) if pd.notna(buf_high) else np.nan,
        "buf_low"     : round(float(buf_low), 3) if pd.notna(buf_low) else np.nan,
        "ATR"         : round(float(atr_val), 3) if pd.notna(atr_val) else np.nan,
    }


# ====== 추가 파라미터 ======
def map_holding_days(h_sel: str) -> int:
    return int(h_sel[1]) if h_sel in {"h1","h2","h3"} else 2


def _to_dateset(dates_like: Optional[List[str]]) -> set:
    if not dates_like:
        return set()
    out = set()
    for s in dates_like:
        try:
            y,m,d = map(int, s.split("-"))
            out.add(date(y,m,d))
        except Exception:
            continue
    return out


def calc_valid_until(holding_days: int, holidays: Optional[List[str]] = None) -> str:
    tz = ZoneInfo("Asia/Seoul") if ZoneInfo else None
    now = datetime.now(tz) if tz else datetime.now()
    target = now + timedelta(days=holding_days)
    hh, mm = DEFAULT_CLOSE_HHMM
    target = target.replace(hour=hh, minute=mm, second=0, microsecond=0)

    holi = _to_dateset(holidays)

    def is_business_day(dt: datetime) -> bool:
        d = dt.date()
        if dt.weekday() >= 5:  # Sat/Sun
            return False
        if d in holi:
            return False
        return True

    # 다음 영업일의 마감시각로 이동
    while not is_business_day(target):
        target += timedelta(days=1)
        target = target.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return target.isoformat()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_confidence(row, h_sel: str) -> float:
    mape = row.get(f"{h_sel}_close_MAPE(%)", np.nan)
    da   = row.get(f"{h_sel}_close_DirAcc(%)", np.nan)
    bias = abs(row.get(f"{h_sel}_close_Bias", np.nan)) if pd.notna(row.get(f"{h_sel}_close_Bias", np.nan)) else np.nan
    mae  = row.get(f"{h_sel}_close_MAE", np.nan)
    cnt  = row.get(f"{h_sel}_close_count", np.nan)

    # 로지스틱 스무딩: 기준 대비 여유가 클수록 1에 가깝게
    # MAPE: 낮을수록 좋음 → (MAPE_MAX - mape) 정규화
    if pd.notna(mape):
        x_m = (MAPE_MAX - mape) / max(MAPE_MAX, 1e-6)
        s_mape = _sigmoid(SIG_K_M * x_m)
    else:
        s_mape = 0.5

    if pd.notna(da):
        x_d = (da - DIRACC_MIN) / max(100 - DIRACC_MIN, 1e-6)
        s_da = _sigmoid(SIG_K_DA * 10 * x_d)  # 0~1 구간 확대
    else:
        s_da = 0.5

    if (pd.notna(bias) and pd.notna(mae) and mae>0):
        ratio = bias / (BIAS_MAE_RATIO_MAX * mae)
        s_bias = _sigmoid( -4.0 * (ratio - 1.0) )  # ratio<1 우수 → >1 급감
    else:
        s_bias = 0.5

    if pd.notna(cnt):
        s_cnt = min(cnt/120.0, 1.0)  # 표본 120 이상은 만점(완만화)
    else:
        s_cnt = 0.5

    conf = (WC_MAPE*s_mape + WC_DA*s_da + WC_BIAS*s_bias + WC_CNT*s_cnt) / (WC_MAPE+WC_DA+WC_BIAS+WC_CNT)
    return round(float(conf), 3)

# ====== 고정 비중 수량 계산 ======
def size_order_qty(entry: float, sl: float,
                   equity: Optional[float],
                   risk_pct: Optional[float],
                   min_qty: int,
                   code: str,
                   tick_fn: TickSizeFn = default_tick_size):
    """고정 비중 방식으로 수량을 계산하는 함수"""
    if not equity or not risk_pct or risk_pct <= 0:
        return max(min_qty, 1), None
    per_share_risk_raw = (entry - sl)
    tick = tick_fn(entry, code)
    per_share_risk = max(per_share_risk_raw, tick)
    # 수수료/슬리피지 추가 반영 (보수적):
    per_share_risk += (entry * FEES_PCT + SLIPPAGE)
    qty = int(np.floor((equity * risk_pct) / max(per_share_risk, 1e-6)))
    return max(qty, min_qty), float(risk_pct)


def decide_side(last_close: float,
                last_pred_close: float,
                buf_close: float,
                diracc: float,
                allow_short: bool=False,
                z_th_long: float=+0.1,
                z_th_short: float=-0.3) -> str:
    """Z-score 기반으로 매수/매도 결정"""
    if pd.isna(last_pred_close) or pd.isna(last_close) or pd.isna(buf_close) or buf_close <= 0:
        return "BUY"  # 정보 부족 시 기본 BUY
    z = (last_pred_close - last_close) / buf_close
    adj_short = z_th_short + (0.5 - (diracc or 50) / 100.0) * 0.2  # DirAcc 낮을수록 숏 관대
    adj_long = z_th_long + ((diracc or 50) / 100.0 - 0.5) * 0.2  # DirAcc 높을수록 롱 관대
    if allow_short and z <= adj_short:
        return "SELL"
    if z >= adj_long:
        return "BUY"
    # 중립대 → 추세 우위로
    return "BUY" if (diracc or 50) >= 55 else ("SELL" if allow_short else "BUY")


# ====== 종목 평가 ======
def evaluate_symbol(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    for t in TARGETS:
        tc = f"true_{t}"
        if tc in df.columns: df[tc] = _safe_num(df[tc])

    wide = {}
    name = df["name"].dropna().iloc[0] if "name" in df.columns else ""
    code = df["code"].dropna().astype(str).str.zfill(6).iloc[0] if "code" in df.columns else ""
    wide["종목명"] = name; wide["종목코드"] = code

    for h in HORIZONS:
        dirres = _diracc_close(df, h)
        for t in TARGETS:
            true_col = f"true_{t}"
            pred_col = f"pred_{h}_{t}"
            if true_col in df.columns and pred_col in df.columns:
                m = _metrics(df[true_col], df[pred_col])
            else:
                m = {"count":0,"RMSE":np.nan,"MAE":np.nan,"MAPE(%)":np.nan,"Bias":np.nan}
            wide[f"{h}_{t}_RMSE"]    = m["RMSE"]
            wide[f"{h}_{t}_MAE"]     = m["MAE"]
            wide[f"{h}_{t}_MAPE(%)"] = m["MAPE(%)"]
            wide[f"{h}_{t}_Bias"]    = m["Bias"]
            wide[f"{h}_{t}_count"]   = m["count"]
        wide[f"{h}_close_DirAcc(%)"] = dirres["DirAcc_close(%)"]
        wide[f"{h}_close_Dir_count"] = dirres["Dir_count"]

    last_true = df["true_close"].dropna().iloc[-1] if "true_close" in df.columns and df["true_close"].notna().any() else np.nan
    if pd.isna(last_true):
        # Fallback 우선순위: 직전 true_close → rolling median → 마지막 pred_close 중 하나
        if "true_close" in df.columns and df["true_close"].notna().sum() >= 2:
            last_true = df["true_close"].dropna().iloc[-2]
        elif "true_close" in df.columns:
            last_true = df["true_close"].rolling(5, min_periods=1).median().iloc[-1]
    wide["last_true_close"] = last_true if pd.notna(last_true) else np.nan

    last_row = df.iloc[-1]
    for h in HORIZONS:
        for t in TARGETS:
            wide[f"last_pred_{h}_{t}"] = last_row.get(f"pred_{h}_{t}", np.nan)

    # ATR 계산
    wide["ATR14"] = _calc_atr(df, ATR_PERIOD)

    return pd.DataFrame([wide])


# ====== 파일별 처리 ======
def process_one_file(csv_path: Path,
                     equity: Optional[float],
                     risk_pct: Optional[float],
                     allow_short: bool,
                     tick_fn: TickSizeFn,
                     holidays: Optional[List[str]]) -> List[dict]:
    try:
        df = pd.read_csv(csv_path, dtype={"code": str})
    except Exception as e:
        logging.warning(f"read fail: {csv_path.name} -> {e}")
        return []

    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    results = []
    if not {"name", "code"}.issubset(df.columns):
        logging.warning(f"skip {csv_path.name}: required columns 'name','code' not found")
        return results

    for (name, code), g in df.groupby(["name", "code"], dropna=False):
        wide = evaluate_symbol(g)
        wrow = wide.iloc[0].to_dict()
        best_h, explain, flags = pick_horizon(wrow)

        last_true = wrow.get("last_true_close", np.nan)
        last_close = last_true if pd.notna(last_true) else wrow.get(f"last_pred_{best_h}_close", np.nan)

        last_pred = {
            "close": wrow.get(f"last_pred_{best_h}_close", np.nan),
            "high":  wrow.get(f"last_pred_{best_h}_high",  np.nan),
            "low":   wrow.get(f"last_pred_{best_h}_low",   np.nan),
        }
        atr_val = wrow.get("ATR14", np.nan)
        lv = compute_levels(wrow, best_h, last_close, last_pred, atr_val, code, tick_fn)

        conf = compute_confidence(wrow, best_h)
        holding_days = map_holding_days(best_h)
        valid_until  = calc_valid_until(holding_days, holidays=holidays)
        qty, risk_used = size_order_qty(lv["매수가(entry)"], lv["손절가(sl)"], equity, risk_pct, min_qty=1, code=code, tick_fn=tick_fn)
        diracc = wrow.get(f"{best_h}_close_DirAcc(%)", np.nan)
        side = decide_side(last_close, last_pred["close"], lv["buf_close"], diracc, allow_short=allow_short)

        res = {
            "source_file": csv_path.name,
            "종목명": name, "종목코드": code,
            "권장호라이즌": best_h,
            "요약점수(h1)": explain["h1"]["total"], "요약점수(h2)": explain["h2"]["total"], "요약점수(h3)": explain["h3"]["total"],
            f"{best_h}_MAPE(%)": wrow.get(f"{best_h}_close_MAPE(%)", np.nan),
            f"{best_h}_DirAcc(%)": wrow.get(f"{best_h}_close_DirAcc(%)", np.nan),
            f"{best_h}_Bias": wrow.get(f"{best_h}_close_Bias", np.nan),
            f"{best_h}_MAE":  wrow.get(f"{best_h}_close_MAE",  np.nan),
            f"{best_h}_count":wrow.get(f"{best_h}_close_count",np.nan),
            "last_close": round(float(last_close), 2) if pd.notna(last_close) else np.nan,
            **lv,
            "ord_qty": int(qty),
            "side": side,
            "confidence": conf,
            "holding_days": holding_days,
            "valid_until": valid_until,
            # 부가정보
            "ex_base_h1": explain["h1"]["base"], "ex_pen_h1": explain["h1"]["penalty"],
            "ex_base_h2": explain["h2"]["base"], "ex_pen_h2": explain["h2"]["penalty"],
            "ex_base_h3": explain["h3"]["base"], "ex_pen_h3": explain["h3"]["penalty"],
            "flags_h1": ";".join(flags["h1"]), "flags_h2": ";".join(flags["h2"]), "flags_h3": ";".join(flags["h3"]),
        }
        if risk_used is not None:
            res["risk_cap_used"] = risk_used

        # RR 비정상 시 경고 플래그
        if not pd.notna(res.get("RR")) or res.get("RR", 0) <= 0:
            res["warn_bad_RR"] = 1
        else:
            res["warn_bad_RR"] = 0

        results.append(res)
    return results

    
def _norm_code(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.str.replace("-", "", regex=False)
    return s.apply(lambda x: x.zfill(6) if x.isdigit() else x)

def make_trade_price(cfg,
                     equity: Optional[float]=None,
                     risk_pct: Optional[float]=None,
                     allow_short: bool=False,
                     tick_fn: TickSizeFn = default_tick_size,
                     holidays: Optional[List[str]] = None) -> pd.DataFrame:

    base_dir = Path(cfg.predict_result_dir) / f"{cfg.end_date}"
    files = sorted(base_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in: {base_dir}")

    all_rows: List[dict] = []
    for fp in files:
        rows = process_one_file(fp, equity=equity, risk_pct=risk_pct,
                                allow_short=allow_short, tick_fn=tick_fn,
                                holidays=holidays)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No results produced from input CSVs.")

    out = pd.DataFrame(all_rows)

    # -------------------------------
    # ① Score_1w 병합 (scored_{end}.csv)
    # -------------------------------
    score_path = Path(cfg.selout_dir) / f"scored_{cfg.end_date}.csv"
    score_df = None
    if score_path.exists():
        # Code 앞자리 0 보존
        score_df = pd.read_csv(score_path, dtype=str)
        # 컬럼 표준화
        score_df = score_df.rename(columns={"Name": "종목명", "Code": "종목코드"})
        # 코드 표준화
        if "종목코드" in score_df.columns:
            score_df["종목코드"] = _norm_code(score_df["종목코드"])
        # 점수 숫자화
        if "Score_1w" in score_df.columns:
            score_df["Score_1w"] = pd.to_numeric(score_df["Score_1w"], errors="coerce")
        else:
            logging.warning("Score file has no 'Score_1w'.")
            score_df["Score_1w"] = np.nan
    else:
        logging.warning(f"Score file not found: {score_path}")

    # out 코드 표준화
    if "종목코드" in out.columns:
        out["종목코드"] = _norm_code(out["종목코드"])

    if score_df is not None and not score_df.empty:
        merged = out.copy()
        merged_done = False

        if "종목코드" in out.columns and "종목코드" in score_df.columns:
            tmp = score_df[["종목코드", "Score_1w"]].drop_duplicates("종목코드")
            merged = merged.merge(tmp, on="종목코드", how="left")
            merged_done = True
        if not merged_done and ("종목명" in out.columns and "종목명" in score_df.columns):
            tmp = score_df[["종목명", "Score_1w"]].drop_duplicates("종목명")
            merged = merged.merge(tmp, on="종목명", how="left")
            merged_done = True

        if not merged_done:
            logging.warning("Cannot merge Score_1w: no common keys ('종목코드' or '종목명').")
        else:
            out = merged

    # -------------------------------
    # ② 컬럼 순서 (질문 예시 구조에 맞춤)
    # -------------------------------
    order = [
        "source_file","종목명","종목코드","권장호라이즌","last_close",
        "매수가(entry)","익절가(tp)","손절가(sl)","RR","Score_1w",
        "ord_qty","side","confidence","holding_days","valid_until",
        "ATR","buf_close","buf_high","buf_low",
        "요약점수(h1)","요약점수(h2)","요약점수(h3)",
        "ex_base_h1","ex_pen_h1","ex_base_h2","ex_pen_h2","ex_base_h3","ex_pen_h3",
        "flags_h1","flags_h2","flags_h3","warn_bad_RR",
    ]
    cols = [c for c in order if c in out.columns] + [c for c in out.columns if c not in order]
    out = out[cols]

    # -------------------------------
    # ③ 정렬: ① Score_1w ↓ → ② RR ↓ → ③ (conf>=0.4 먼저) → confidence ↓
    # -------------------------------
    if "Score_1w" not in out.columns:
        out["Score_1w"] = np.nan
    out["RR"] = pd.to_numeric(out.get("RR", np.nan), errors="coerce")
    out["confidence"] = pd.to_numeric(out.get("confidence", np.nan), errors="coerce")

    out["_conf_bucket"] = (out["confidence"].fillna(-np.inf) >= 0.4).astype(int)
    out["_confidence_num"] = out["confidence"]

    out.sort_values(
        by=["Score_1w", "RR", "_conf_bucket", "_confidence_num"],
        ascending=[False, False, False, False],
        na_position="last",
        inplace=True
    )
    out.drop(columns=["_conf_bucket","_confidence_num"], inplace=True)

    # -------------------------------
    # ④ 저장
    # -------------------------------
    report_dir = Path(cfg.price_report_dir) / f"Report_{cfg.end_date}"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"Trading_price_{cfg.end_date}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    logging.info(f"saved -> {out_path}")

    return out

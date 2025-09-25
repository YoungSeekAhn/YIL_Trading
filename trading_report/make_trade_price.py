# -*- coding: utf-8 -*-
"""
batch_select_horizon_and_levels_v3.py
- <predict_dir>/<end_date> 폴더의 *.csv 를 '각 파일별로' 독립 처리
- 파일마다: (여러 종목이 있을 수 있음) → 종목별 평가/권장호라이즌/레벨/추가파라미터 생성
- 모든 파일의 결과를 리스트로 모아 마지막에 단일 CSV로 저장

v2와 차이:
  - 더 이상 입력 CSV들을 먼저 concat 하지 않음
  - 파일 단위로 처리 후 결과만 누적

CLI 예:
  python batch_select_horizon_and_levels_v3.py --predict-dir ./predict --end-date 20250925 \
      --out ./recommended_levels_v3.csv --equity 50000000 --risk-pct 0.005 --allow-short
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# ====== 설정 ======
HORIZONS = ["h1","h2","h3"]
TARGETS  = ["high","low","close"]
EPS = 1e-12

# 권장 기준(보수적)
MAPE_MAX = 2.5
DIRACC_MIN = 55.0
BIAS_MAE_RATIO_MAX = 0.5
COUNT_MIN = 40

# 점수 가중치(호라이즌 선택)
W_MAPE = 0.40
W_DA   = 0.30
W_BIAS = 0.20
W_MAE  = 0.10

# 트레이딩 파라미터
RISK_REWARD    = 1.8
ENTRY_PULLBACK = 0.25  # entry = last_close - 0.25*buf_close

# 신뢰도 가중치 (0~1 스케일)
WC_MAPE = 0.40
WC_DA   = 0.30
WC_BIAS = 0.20
WC_CNT  = 0.10

# 마감 시각(현물)
DEFAULT_CLOSE_HHMM = (15, 20)  # 15:20 KST


# ====== 유틸 ======
def _safe_num(s): return pd.to_numeric(s, errors="coerce")

def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = _safe_num(y_true); y_pred = _safe_num(y_pred)
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"count":0,"RMSE":np.nan,"MAE":np.nan,"MAPE(%)":np.nan,"Bias":np.nan}
    e  = (y_pred[mask] - y_true[mask]).values
    yt = y_true[mask].values
    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e / np.clip(yt, EPS, None))) * 100.0)
    bias = float(np.mean(e))
    return {"count": int(mask.sum()), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias}

def _diracc_close(df: pd.DataFrame, h: str) -> Dict[str, float]:
    if "true_close" not in df.columns: return {"DirAcc_close(%)": np.nan, "Dir_count":0}
    pcol = f"pred_{h}_close"
    if pcol not in df.columns: return {"DirAcc_close(%)": np.nan, "Dir_count":0}
    df = df.sort_values("date")
    y_true = _safe_num(df["true_close"])
    y_pred = _safe_num(df[pcol])
    prev_true = y_true.shift(1)
    mask = y_true.notna() & y_pred.notna() & prev_true.notna()
    if mask.sum() == 0: return {"DirAcc_close(%)": np.nan, "Dir_count":0}
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
        "DA":   row.get(f"{h}_close_DirAcc(%)", np.nan),
        "BiasAbs": abs(row.get(f"{h}_close_Bias", np.nan)) if pd.notna(row.get(f"{h}_close_Bias", np.nan)) else np.nan,
        "MAE":  row.get(f"{h}_close_MAE", np.nan),
        "Count":row.get(f"{h}_close_count", np.nan)
    }

def pick_horizon(row) -> Tuple[str, Dict]:
    hs = HORIZONS
    blocks = {h:_score_hblock(row, h) for h in hs}
    mape_n = _rank_norm([blocks[h]["MAPE"] for h in hs], ascending=True)
    da_n   = _rank_norm([blocks[h]["DA"]   for h in hs], ascending=False)
    bias_n = _rank_norm([blocks[h]["BiasAbs"] for h in hs], ascending=True)
    mae_n  = _rank_norm([blocks[h]["MAE"]     for h in hs], ascending=True)
    base = {h: (W_MAPE*mape_n[i] + W_DA*da_n[i] + W_BIAS*bias_n[i] + W_MAE*mae_n[i]) for i,h in enumerate(hs)}
    pen  = {h: 0.0 for h in hs}
    for h in hs:
        b = blocks[h]
        if pd.notna(b["MAPE"]) and b["MAPE"] > MAPE_MAX: pen[h] += 0.50
        if pd.notna(b["DA"])   and b["DA"]   < DIRACC_MIN: pen[h] += 0.25
        if pd.notna(b["MAE"]) and pd.notna(b["BiasAbs"]) and (b["BiasAbs"] > BIAS_MAE_RATIO_MAX * b["MAE"]): pen[h] += 0.20
        if pd.notna(b["Count"]) and b["Count"] < COUNT_MIN: pen[h] += 0.25
    total = {h: base[h] + pen[h] for h in hs}
    best = sorted(hs, key=lambda x: (
        total[x],
        blocks[x]["MAPE"] if pd.notna(blocks[x]["MAPE"]) else np.inf,
        -(blocks[x]["DA"] if pd.notna(blocks[x]["DA"]) else -np.inf)
    ))[0]
    explain = {h: {"base": round(base[h],4), "penalty": round(pen[h],4), "total": round(total[h],4)} for h in hs}
    return best, explain

def _buf(mae, bias):
    vals = [mae, abs(bias)]
    vals = [v for v in vals if pd.notna(v)]
    return max(vals) if vals else np.nan

def compute_levels(row, h_sel, last_close, last_pred: Dict[str, float]):
    mae = row.get(f"{h_sel}_close_MAE", np.nan)
    bias= row.get(f"{h_sel}_close_Bias", np.nan)
    mae_h= row.get(f"{h_sel}_high_MAE", np.nan); bias_h = row.get(f"{h_sel}_high_Bias", np.nan)
    mae_l= row.get(f"{h_sel}_low_MAE",  np.nan); bias_l = row.get(f"{h_sel}_low_Bias",  np.nan)
    buf_close = _buf(mae, bias)
    buf_high  = _buf(mae_h, bias_h) if pd.notna(mae_h) or pd.notna(bias_h) else buf_close
    buf_low   = _buf(mae_l, bias_l) if pd.notna(mae_l) or pd.notna(bias_l) else buf_close
    if pd.isna(buf_close): buf_close = max(1.0, mae if pd.notna(mae) else 1.0)

    entry = float(last_close) - ENTRY_PULLBACK * buf_close
    risk  = buf_close
    tp    = float(last_close) + RISK_REWARD * risk
    sl    = float(last_close) - risk

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

    rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else np.nan
    return {
        "매수가(entry)": round(entry, 2),
        "익절가(tp)"   : round(tp, 2),
        "손절가(sl)"   : round(sl, 2),
        "RR"          : round(rr, 2),
        "buf_close"   : round(buf_close, 2)
    }

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
    wide["last_true_close"] = last_true
    last_row = df.iloc[-1]
    for h in HORIZONS:
        for t in TARGETS:
            wide[f"last_pred_{h}_{t}"] = last_row.get(f"pred_{h}_{t}", np.nan)

    return pd.DataFrame([wide])

# ====== 추가 파라미터 ======
def map_holding_days(h_sel: str) -> int:
    return int(h_sel[1]) if h_sel in {"h1","h2","h3"} else 2

def calc_valid_until(holding_days: int) -> str:
    tz = ZoneInfo("Asia/Seoul") if ZoneInfo else None
    now = datetime.now(tz) if tz else datetime.now()
    target = now + timedelta(days=holding_days)
    hh, mm = DEFAULT_CLOSE_HHMM
    target = target.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target.weekday() == 5:   # Sat
        target += timedelta(days=2)
    elif target.weekday() == 6: # Sun
        target += timedelta(days=1)
        target = target.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return target.isoformat()

def compute_confidence(row, h_sel: str) -> float:
    mape = row.get(f"{h_sel}_close_MAPE(%)", np.nan)
    da   = row.get(f"{h_sel}_close_DirAcc(%)", np.nan)
    bias = abs(row.get(f"{h_sel}_close_Bias", np.nan)) if pd.notna(row.get(f"{h_sel}_close_Bias", np.nan)) else np.nan
    mae  = row.get(f"{h_sel}_close_MAE", np.nan)
    cnt  = row.get(f"{h_sel}_close_count", np.nan)
    s_mape = 1 - min((mape or np.inf)/MAPE_MAX, 1) if pd.notna(mape) else 0.5
    s_da   = min(max(((da or 0) - 50) / (DIRACC_MIN - 50 if DIRACC_MIN>50 else 5), 0), 1) if pd.notna(da) else 0.5
    s_bias = max(1 - (bias / (BIAS_MAE_RATIO_MAX * mae)) , 0) if (pd.notna(bias) and pd.notna(mae) and mae>0) else 0.5
    s_cnt  = min((cnt or 0)/60.0, 1) if pd.notna(cnt) else 0.5
    conf = (WC_MAPE*s_mape + WC_DA*s_da + WC_BIAS*s_bias + WC_CNT*s_cnt) / (WC_MAPE+WC_DA+WC_BIAS+WC_CNT)
    return round(float(conf), 3)

def size_order_qty(entry: float, sl: float,
                   equity: Optional[float],
                   risk_pct: Optional[float],
                   min_qty: int = 1):
    if not equity or not risk_pct or risk_pct <= 0:
        return max(min_qty, 1), None
    per_share_risk = max(entry - sl, 1.0)
    qty = int(np.floor((equity * risk_pct) / per_share_risk))
    return max(qty, min_qty), float(risk_pct)

def decide_side(last_close: float, last_pred_close: float, buf_close: float, allow_short: bool=False) -> str:
    if allow_short and (pd.notna(last_pred_close) and pd.notna(last_close) and pd.notna(buf_close)):
        if last_pred_close <= (last_close - 0.8*buf_close):
            return "SELL"
    return "BUY"


# ====== 핵심: 파일별 처리 후 누적 ======
def process_one_file(csv_path: Path,
                     equity: Optional[float],
                     risk_pct: Optional[float],
                     allow_short: bool) -> List[dict]:
    """단일 CSV 파일을 처리하여 결과 행(dict) 목록 반환"""
    try:
        df = pd.read_csv(csv_path, dtype={"code": str})
    except Exception as e:
        print(f"[WARN] read fail: {csv_path.name} -> {e}")
        return []

    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    results = []
    # 파일 안에 여러 종목이 들어 있을 수 있으므로 groupby
    if not {"name","code"}.issubset(df.columns):
        print(f"[WARN] skip {csv_path.name}: required columns 'name','code' not found")
        return results

    for (name, code), g in df.groupby(["name","code"], dropna=False):
        wide = evaluate_symbol(g)
        wrow = wide.iloc[0].to_dict()
        best_h, explain = pick_horizon(wrow)

        last_true = wrow.get("last_true_close", np.nan)
        if pd.notna(last_true):
            last_close = last_true
        else:
            last_close = wrow.get(f"last_pred_{best_h}_close", np.nan)

        last_pred = {
            "close": wrow.get(f"last_pred_{best_h}_close", np.nan),
            "high":  wrow.get(f"last_pred_{best_h}_high",  np.nan),
            "low":   wrow.get(f"last_pred_{best_h}_low",   np.nan),
        }
        lv = compute_levels(wrow, best_h, last_close, last_pred)

        conf = compute_confidence(wrow, best_h)
        holding_days = map_holding_days(best_h)
        valid_until  = calc_valid_until(holding_days)
        qty, risk_used = size_order_qty(lv["매수가(entry)"], lv["손절가(sl)"], equity, risk_pct, min_qty=1)
        side = decide_side(last_close, last_pred["close"], lv["buf_close"], allow_short=allow_short)

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
            "valid_until": valid_until
        }
        if risk_used is not None:
            res["risk_cap_used"] = risk_used
        results.append(res)
    return results


def make_trade_price(cfg, equity: Optional[float]=None, risk_pct: Optional[float]=None,
                 allow_short: bool=False) -> pd.DataFrame:
    
    base_dir = Path(cfg.predict_dir) / f"{cfg.end_date}"
    files = sorted(base_dir.glob("*.csv"))
    
    if not files:
        raise FileNotFoundError(f"No CSV found in: {base_dir}")

    all_rows: List[dict] = []
    for fp in files:
        rows = process_one_file(fp, equity=equity, risk_pct=risk_pct, allow_short=allow_short)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No results produced from input CSVs.")

    out = pd.DataFrame(all_rows)

    # 보기 좋은 컬럼 순서
    order = ["source_file","종목명","종목코드","권장호라이즌","last_close",
             "매수가(entry)","익절가(tp)","손절가(sl)","RR",
             "ord_qty","side","confidence","holding_days","valid_until",
             "요약점수(h1)","요약점수(h2)","요약점수(h3)",
             "h1_close_MAPE(%)","h2_close_MAPE(%)","h3_close_MAPE(%)",
             "h1_close_DirAcc(%)","h2_close_DirAcc(%)","h3_close_DirAcc(%)"]
    cols = [c for c in order if c in out.columns] + [c for c in out.columns if c not in order]
    out = out[cols]

    out_path = Path(cfg.report_dir) / f"Report_{cfg.end_date}" / f"Trading_price_{cfg.end_date}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] saved -> {out_path}")
    return out

   #--equity", type=float, default=None, help="계좌 총자산(원). 리스크 기반 수량 산정에 사용")
   #"--risk-pct", type=float, default=None, help="트레이드당 허용 위험 비율(예: 0.005=0.5%)")
   #"--allow-short", action="store_true", help="하락 예측 시 SELL 허용")

import pandas as pd
import numpy as np
from pathlib import Path
from makedata.dataset_functions import last_trading_day

import os   

from DSConfig_3 import DSConfig

cfg = DSConfig
cfg.end_date = last_trading_day()

# -----------------------------
# 사용자 설정
# -----------------------------
CSV_PATH = cfg.report_dir    # 질문 형식의 wide CSV (여러 종목)
H = "h2"                             # 사용할 호라이즌: 'h1' | 'h2' | 'h3'
MAPE_MAX = 2.5                       # % 단위, 종가 MAPE 기준
DIRACC_MIN = 55.0                    # % 단위, 종가 방향정확도 기준
BIAS_MAE_RATIO_MAX = 0.5             # |Bias| <= 0.5 * MAE
COUNT_MIN = 40                       # 종가 count

RISK_REWARD = 1.8                    # RR = 기대이익/위험, 1.5~2.0 권장
ENTRY_PULLBACK = 0.25                # 매수가를 last_close - 0.25*buf_close로 약간 유리하게

"""
select_horizon_and_trade_levels.py
- Predic_result.csv (wide) 로부터 종목별 권장 호라이즌(h1/h2/h3) 선정
- 선정 호라이즌으로 매수가/익절가/손절가 산출
"""

import pandas as pd
import numpy as np
from pathlib import Path
#from makedata.dataset_functions import last_trading_day

import os   

from DSConfig_3 import DSConfig

cfg = DSConfig
cfg.end_date = last_trading_day()

# -----------------------------
# 사용자 설정
# -----------------------------
# 최신 종가(또는 체결가) 주입 (실거래 필수). 없으면 데모 스케일로 임시 산정.
latest_close = {
    # "34730": 170000,
    # "18260": 150000,
}

# ---------- 기준(보수적 권고) ----------
MAPE_MAX = 2.5          # %
DIRACC_MIN = 55.0       # %
BIAS_MAE_RATIO_MAX = 0.5
COUNT_MIN = 40          # 표본 최소

# ---------- 점수 가중치 ----------
W_MAPE = 0.40
W_DA   = 0.30   # DirAcc는 높을수록 좋음(내부에서 역정규화 처리)
W_BIAS = 0.20   # |Bias|
W_MAE  = 0.10

# ---------- 트레이딩 파라미터 ----------
RISK_REWARD = 1.8      # TP/SL 비율
ENTRY_PULLBACK = 0.25  # entry = last_close - 0.25*buf_close

def rank_norm(vals, ascending=True):
    """작을수록 좋은 방향으로 [0,1] 정규화(0=최상, 1=최하). NaN은 중간값 대체."""
    s = pd.Series(vals, dtype=float)
    # 모두 NaN이면 0.5 반환
    if s.dropna().empty:
        return [0.5] * len(s)
    r = s.rank(ascending=ascending, method="average")
    # 1..k -> 0..1
    norm = (r - 1) / max(len(s) - 1, 1)
    # NaN을 중앙값으로
    norm[s.isna()] = 0.5
    return norm.tolist()

def horizon_score_block(row, h):
    """한 종목-한 호라이즌의 스코어 및 페널티 계산"""
    mape = row.get(f"{h}_close_MAPE(%)", np.nan)
    da   = row.get(f"{h}_close_DirAcc(%)", np.nan)
    bias = abs(row.get(f"{h}_close_Bias", np.nan))
    mae  = row.get(f"{h}_close_MAE", np.nan)
    cnt  = row.get(f"{h}_close_count", np.nan)
    return {"MAPE": mape, "DA": da, "BiasAbs": bias, "MAE": mae, "Count": cnt}

def pick_recommended_horizon(row):
    """h1/h2/h3의 상대 비교(랭크 정규화 + 페널티)로 권장 호라이즌 선정"""
    hs = ["h1", "h2", "h3"]
    blocks = {h: horizon_score_block(row, h) for h in hs}
    # 정규화(호라이즌 간 상대비교)
    mape_norm = rank_norm([blocks[h]["MAPE"] for h in hs], ascending=True)   # 낮을수록 좋음
    da_norm   = rank_norm([blocks[h]["DA"]   for h in hs], ascending=False)  # 높을수록 좋음 → 역정규화(낮을수록 좋게)
    bias_norm = rank_norm([blocks[h]["BiasAbs"] for h in hs], ascending=True)
    mae_norm  = rank_norm([blocks[h]["MAE"]     for h in hs], ascending=True)

    base_scores = {
        h: (W_MAPE*mape_norm[i] + W_DA*da_norm[i] + W_BIAS*bias_norm[i] + W_MAE*mae_norm[i])
        for i, h in enumerate(hs)
    }

    # 페널티(절대 기준 미충족 시 가점)
    penalties = {h: 0.0 for h in hs}
    for i, h in enumerate(hs):
        b = blocks[h]
        if pd.notna(b["MAPE"]) and b["MAPE"] > MAPE_MAX:             penalties[h] += 0.50
        if pd.notna(b["DA"])   and b["DA"]   < DIRACC_MIN:           penalties[h] += 0.25
        if pd.notna(b["MAE"]) and pd.notna(b["BiasAbs"]) and (b["BiasAbs"] > BIAS_MAE_RATIO_MAX * b["MAE"]):
            penalties[h] += 0.20
        if pd.notna(b["Count"]) and b["Count"] < COUNT_MIN:          penalties[h] += 0.25

    total = {h: base_scores[h] + penalties[h] for h in hs}
    # 최저 점수 선택 (tie-break: MAPE → DA 내림차순)
    best = sorted(
        hs,
        key=lambda x: (
            total[x],
            blocks[x]["MAPE"] if pd.notna(blocks[x]["MAPE"]) else np.inf,
            -(blocks[x]["DA"] if pd.notna(blocks[x]["DA"]) else -np.inf)
        )
    )[0]

    # 디버그/설명용 정보 반환
    explain = {
        h: {
            "base": round(base_scores[h], 4),
            "penalty": round(penalties[h], 4),
            "total": round(total[h], 4),
            "MAPE": blocks[h]["MAPE"],
            "DA": blocks[h]["DA"],
            "BiasAbs": blocks[h]["BiasAbs"],
            "MAE": blocks[h]["MAE"],
            "Count": blocks[h]["Count"],
        } for h in hs
    }
    return best, explain

def compute_levels(row, H_sel, last_close, pred=None):
    """
    집계 지표만으로 보수적 entry/tp/sl 산출.
    pred: {'close':..., 'high':..., 'low':...} optional (해당 호라이즌의 예측치가 있을 때)
    """
    def _buf(mae, bias):
        vals = [mae, abs(bias)]
        vals = [v for v in vals if pd.notna(v)]
        return max(vals) if vals else np.nan

    buf_close = _buf(row[f"{H_sel}_close_MAE"], row[f"{H_sel}_close_Bias"])
    buf_high  = _buf(row.get(f"{H_sel}_high_MAE"), row.get(f"{H_sel}_high_Bias"))
    buf_low   = _buf(row.get(f"{H_sel}_low_MAE"),  row.get(f"{H_sel}_low_Bias"))

    # 누락 시 대체
    if pd.isna(buf_close): buf_close = max(1.0, row[f"{H_sel}_close_MAE"])
    if pd.isna(buf_high):  buf_high  = buf_close
    if pd.isna(buf_low):   buf_low   = buf_close

    entry = last_close - ENTRY_PULLBACK * buf_close
    risk  = buf_close
    tp    = last_close + RISK_REWARD * risk
    sl    = last_close - risk

    if pred:
        p_close = pred.get("close", np.nan)
        p_high  = pred.get("high",  np.nan)
        p_low   = pred.get("low",   np.nan)
        # TP: 예측 상방이 있으면 그 아래로 여유
        if not pd.isna(p_high) or not pd.isna(p_close):
            best_up = np.nanmin([p for p in [p_high, p_close] if not pd.isna(p)]) if any([not pd.isna(p_high), not pd.isna(p_close)]) else p_close
            if not pd.isna(best_up):
                tp = min(tp, best_up - 0.5*buf_high)
        # SL: 예측 저점이 있으면 그 위로 여유
        if not pd.isna(p_low):
            sl = max(sl, p_low + 0.5*buf_low)

    rr = (tp - entry) / (entry - sl) if (entry - sl) > 0 else np.nan

    return {
        "entry": round(entry, 2), "tp": round(tp, 2), "sl": round(sl, 2), "RR": round(rr, 2),
        "buf_close": round(buf_close, 2), "buf_high": round(buf_high, 2), "buf_low": round(buf_low, 2)
    }

def make_trade_price(cfg):
    # ------------------- 메인 파이프라인 -------------------
    file_path = Path(cfg.report_dir) / f"Report_{cfg.end_date}"
    file_path = file_path / f"Predict_result_{cfg.end_date}.csv"

    df = pd.read_csv(file_path, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].str.zfill(6)
    for c in df.columns:
        if c not in ["종목명", "종목코드"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    reco_rows = []
    for _, row in df.iterrows():
        best_h, explain = pick_recommended_horizon(row)

        code = row["종목코드"]; name = row["종목명"]
        lc = latest_close.get(code)
        if lc is None:
            # 데모 스케일: MAE*100 (실거래에서는 반드시 최신 종가로 대체)
            approx = row.get(f"{best_h}_close_MAE", 1000.0)
            lc = max(approx, 1000.0) * 100.0

        levels = compute_levels(row, best_h, lc, pred=None)

        # 요약 행
        reco_rows.append({
            "종목명": name, "종목코드": code,
            "권장호라이즌": best_h,
            "요약점수(h1)": explain["h1"]["total"], "요약점수(h2)": explain["h2"]["total"], "요약점수(h3)": explain["h3"]["total"],
            f"{best_h}_MAPE(%)": row[f"{best_h}_close_MAPE(%)"],
            f"{best_h}_DirAcc(%)": row[f"{best_h}_close_DirAcc(%)"],
            f"{best_h}_Bias": row[f"{best_h}_close_Bias"],
            f"{best_h}_MAE": row[f"{best_h}_close_MAE"],
            f"{best_h}_count": row[f"{best_h}_close_count"],
            "last_close": round(lc, 2),
            "매수가(entry)": levels["entry"],
            "익절가(tp)": levels["tp"],
            "손절가(sl)": levels["sl"],
            "RR": levels["RR"]
        })

    result = pd.DataFrame(reco_rows)
    # 가독성 정렬: 권장호라이즌 점수가 낮을수록 상단
    result = result.sort_values(["권장호라이즌", "RR"], ascending=[True, False]).reset_index(drop=True)

    print("=== 권장 호라이즌 및 트레이드 레벨 ===")
    cols = ["종목명","종목코드","권장호라이즌",
            "last_close","매수가(entry)","익절가(tp)","손절가(sl)","RR",
            "요약점수(h1)","요약점수(h2)","요약점수(h3)",
            "h1_Med","h2_Med","h3_Med"]
    # 출력 컬럼 보정 (호라이즌별 메트릭은 동적으로 존재하므로 안전 처리)
    print(
        result[
            [c for c in ["종목명","종목코드","권장호라이즌","last_close","매수가(entry)","익절가(tp)","손절가(sl)","RR",
                        "요약점수(h1)","요약점수(h2)","요약점수(h3)",
                        "h1_close_MAPE(%)","h2_close_MAPE(%)","h3_close_MAPE(%)",
                        "h1_close_DirAcc(%)","h2_close_DirAcc(%)","h3_close_DirAcc(%)"] if c in result.columns]
        ].to_string(index=False)
    )

    # 저장(선택)
    out_path = Path(cfg.report_dir) / f"Report_{cfg.end_date}" / f"Trading_price_{cfg.end_date}.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    # result.to_csv("./recommended_horizon_and_levels.csv", index=False, encoding="utf-8-sig")

file_path = Path(cfg.predict_dir) / f"{cfg.end_date}"

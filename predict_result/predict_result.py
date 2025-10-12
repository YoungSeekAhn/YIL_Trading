# 1️⃣ 지표 설명 복습
# RMSE (Root Mean Squared Error): 오차 제곱 평균의 제곱근, 큰 오차에 민감.
# MAE (Mean Absolute Error): 절대 오차 평균, 직관적인 “평균 차이”.
# MAPE(%): 실제 대비 상대 오차 비율(%).
# Bias: 평균 오차(양수 → 과대예측, 음수 → 과소예측).
# count: 평가에 사용된 유효 샘플 수.
# DirAcc(%): 방향성 적중률(상승/하락 맞춘 비율).


import os, glob, re
import pandas as pd
import numpy as np
from pathlib import Path
from DSConfig_3 import DSConfig, config
from makedata.dataset_functions import last_trading_day


HORIZONS = [1, 2, 3]
TARGETS  = ["high", "low", "close"]  # 평가 대상
EPS = 1e-12

# ---------------------------
# 종목명/코드 추출
# ---------------------------
def parse_symbol_from_filename(path: str):
    """
    파일명에서 (종목코드, 종목명)을 추출.
    허용 예:
      - '005930_삼성전자.csv'
      - '삼성전자_005930.csv'
      - '005930.csv' or '삼성전자.csv'
    우선순위: 숫자 6자리 -> 종목코드로 간주
    """
    base = os.path.splitext(os.path.basename(path))[0]
    parts = re.split(r'[_\- ]+', base)

    code = None
    name = None
    for p in parts:
        if re.fullmatch(r'\d{6}', p):
            code = p
        else:
            # 한글/영문 혼재 종목명일 수 있음
            name = p if name is None else name

    # 둘 다 없으면 전체를 이름으로
    if code is None and name is None:
        name = base
    return name or "", code or ""

# ---------------------------
# 메트릭/방향 적중률
# ---------------------------
def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _metrics(y_true, y_pred):
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return {"count": 0, "RMSE": np.nan, "MAE": np.nan, "MAPE(%)": np.nan, "Bias": np.nan}
    e = (y_pred[mask] - y_true[mask]).values
    yt = y_true[mask].values
    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))
    mape = float(np.mean(np.abs(e / np.clip(yt, EPS, None))) * 100.0)
    bias = float(np.mean(e))
    return {"count": int(mask.sum()), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias}

def _direction_accuracy_close(df: pd.DataFrame, h: int):
    """
    방향 적중률: close 기준
    sign(pred_h{h}_close - prev_true_close) == sign(true_close - prev_true_close)
    """
    col_pred = f"pred_h{h}_close"
    if "true_close" not in df.columns or col_pred not in df.columns:
        return {"DirAcc_close(%)": np.nan, "Dir_count": 0}

    df = df.sort_values("date")
    y_true = pd.to_numeric(df["true_close"], errors="coerce")
    y_pred = pd.to_numeric(df[col_pred], errors="coerce")
    prev_true = y_true.shift(1)

    mask = y_true.notna() & y_pred.notna() & prev_true.notna()
    if mask.sum() == 0:
        return {"DirAcc_close(%)": np.nan, "Dir_count": 0}

    true_dir = np.sign(y_true[mask] - prev_true[mask])
    pred_dir = np.sign(y_pred[mask] - prev_true[mask])
    acc = float((true_dir.values == pred_dir.values).mean() * 100.0)
    return {"DirAcc_close(%)": acc, "Dir_count": int(mask.sum())}

# ---------------------------
# 단일 종목 -> 1행 wide 포맷
# ---------------------------
def evaluate_symbol_to_wide(df: pd.DataFrame, stock_name: str, stock_code: str):
    # 날짜/정렬 & 형 변환
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    for t in ["high", "low", "close"]:
        col = f"true_{t}"
        if col in df.columns:
            df[col] = _safe_num(df[col])

    for h in HORIZONS:
        for t in TARGETS:
            col = f"pred_h{h}_{t}"
            if col in df.columns:
                df[col] = _safe_num(df[col])

    row = {"종목명": stock_name, "종목코드": stock_code}

    # 각 호라이즌/타깃별 메트릭
    for h in HORIZONS:
        # 방향 적중률 (close만)
        dir_res = _direction_accuracy_close(df, h)

        for t in TARGETS:
            true_col = f"true_{t}"
            pred_col = f"pred_h{h}_{t}"
            if true_col not in df.columns or pred_col not in df.columns:
                # 없는 컬럼은 NaN으로 채움
                row[f"h{h}_{t}_RMSE"]   = np.nan
                row[f"h{h}_{t}_MAE"]    = np.nan
                row[f"h{h}_{t}_MAPE(%)"]= np.nan
                row[f"h{h}_{t}_Bias"]   = np.nan
                row[f"h{h}_{t}_count"]  = 0
            else:
                m = _metrics(df[true_col], df[pred_col])
                row[f"h{h}_{t}_RMSE"]    = m["RMSE"]
                row[f"h{h}_{t}_MAE"]     = m["MAE"]
                row[f"h{h}_{t}_MAPE(%)"] = m["MAPE(%)"]
                row[f"h{h}_{t}_Bias"]    = m["Bias"]
                row[f"h{h}_{t}_count"]   = m["count"]

        # close 방향성
        row[f"h{h}_close_DirAcc(%)"]  = dir_res["DirAcc_close(%)"]
        row[f"h{h}_close_Dir_count"]  = dir_res["Dir_count"]

    return pd.DataFrame([row])

# ---------------------------
# 폴더 단위 일괄 처리
# ---------------------------
def predict_result(cfg):
    
    input_dir = Path(cfg.predict_result_dir) / f"{cfg.end_date}"
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No files in {input_dir}/*.csv")

    all_rows = []

    for path in paths:
        df = pd.read_csv(path)

        # 파일 내 Name/Code 우선 사용
        stock_name = ""
        stock_code = ""
        name_col = next((c for c in df.columns if c.lower() in ["name","종목명"]), None)
        code_col = next((c for c in df.columns if c.lower() in ["code","종목코드"]), None)
        if name_col: stock_name = str(df[name_col].iloc[0])
        if code_col: stock_code = str(df[code_col].iloc[0])

        wide = evaluate_symbol_to_wide(df, stock_name, stock_code)
        all_rows.append(wide)

    out = pd.concat(all_rows, ignore_index=True)

    # 컬럼 정렬: 종목명, 종목코드 먼저
    cols_front = ["종목명", "종목코드"]
    other_cols = [c for c in out.columns if c not in cols_front]
    out = out[cols_front + other_cols]

    output_dir = Path(cfg.predict_report_dir); output_dir.mkdir(exist_ok=True, parents=True)
    output_csv = output_dir /f"Report_{cfg.end_date}" / f"Predict_result_{cfg.end_date}.csv"
    output_csv.parent.mkdir(exist_ok=True, parents=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    return out

# ---------------------------
# 사용 예
# ---------------------------
if __name__ == "__main__":
    # 예: ./pred_results 폴더에 종목별 CSV가 있을 때
    # 결과: symbol_metrics_wide.csv (종목당 1행)
    out = evaluate_predict_result()
    print(out.head())
    pass

# plot_test_prices_aligned.py
# ------------------------------------------------------------
# 저장된 모델과 테스트로더를 이용해
# (N,H,3=[high,low,close]) 멀티호라이즌 예측을
# 목표일에 정렬(Aligned)하여 CSV/플롯 생성
# - target_kind='close3_logr' 지원 (절대가 복원)
# - artifacts/scalers.pkl 로드하여 price 브랜치 역스케일
# ------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import BDay
import pickle

from DSConfig_3 import DSConfig, FeatureConfig

# ========== 설정 ==========
BATCH_SIZE = 128
ROLLING_H1_PLOT = True   # h=1 연속 경로(채널별)도 그림
CHANNELS = ["high", "low", "close"]  # Y[...,0/1/2] 채널명 (순서 고정)
# =========================

feature = FeatureConfig()  # 필요 시 사용 (여기선 칼럼명 참조만 가능)

# ---------------- 유틸 ----------------
def _ensure_dtindex(arr_like):
    idx = pd.DatetimeIndex(pd.to_datetime(arr_like, errors="coerce")).tz_localize(None)
    if idx.isna().any() or len(idx) == 0:
        raise ValueError("date 파싱 실패 또는 비어 있음.")
    return idx

def _price_indices_from_scaler_pack(price_pack: dict) -> Dict[str, int]:
    """
    scalers.pkl의 price 팩에서 high/low/close의 컬럼 인덱스를 찾음.
    반환: {'high': i, 'low': j, 'close': k}
    """
    price_cols = list(price_pack.get("cols", []))
    idx = {}
    for ch in CHANNELS:
        if ch in price_cols:
            idx[ch] = price_cols.index(ch)
        else:
            # 못 찾으면 기본값 폴백 (open,high,low,close,volume... 가정 시 close=3 등)
            # 안전하게 KeyError 유도
            raise KeyError(f"[price scaler cols]에 '{ch}'가 없습니다. cols={price_cols}")
    return idx

# ---------------- 정렬 배열 생성 (멀티채널, 절대가 복원) ----------------
def build_aligned_series_from_base_multich(
    preds: np.ndarray,      # (N, H, 3) : 로그수익률(r)
    trues: np.ndarray,      # (N, H, 3) : 로그수익률(r)
    base_prices: np.ndarray, # (N, 3)    : [H,L,C] 기준 절대가
    horizons: Union[List[int], Tuple[int, ...]],
    target_kind: str = "close3_logr",
):
    """
    반환:
      aligned_pred: dict[h][ch] -> 1D array (len=N+h_max)
      aligned_true: dict[h][ch] -> 1D array (len=N+h_max)

    close3_logr:
      절대가 = base_price * exp(r)
      (여기서는 단일 스텝(+h)에 대한 수준 예측을 만든 뒤, 목표일에 정렬만 수행)
    """
    if not isinstance(horizons, (list, tuple)):
        horizons = [int(horizons)]
    Hs = list(map(int, horizons))

    assert preds.ndim == 3 and trues.ndim == 3, f"preds,trues must be (N,H,3). got {preds.shape=} {trues.shape=}"
    assert preds.shape == trues.shape, "preds/trues shape mismatch"
    N, H, C = preds.shape
    assert C == 3, f"last dim must be 3 (high,low,close). got {C}"
    assert base_prices.shape == (N, 3), f"base_prices must be (N,3). got {base_prices.shape}"

    h_max = max(Hs)
    aligned_pred = {h: {ch: np.full(N + h_max, np.nan, dtype=float) for ch in CHANNELS} for h in Hs}
    aligned_true = {h: {ch: np.full(N + h_max, np.nan, dtype=float) for ch in CHANNELS} for h in Hs}

    if target_kind != "close3_logr":
        raise NotImplementedError("현재는 target_kind='close3_logr'만 지원합니다.")

    # 성장률(g) = exp(r), 절대가 = base * g
    g_pred = np.exp(preds)  # (N,H,3)
    g_true = np.exp(trues)  # (N,H,3)

    for j, h in enumerate(Hs):
        if j >= H:
            break
        for k in range(N):
            t = k + h
            if t < N + h_max:
                for c, ch in enumerate(CHANNELS):
                    aligned_pred[h][ch][t] = float(base_prices[k, c]) * float(g_pred[k, j, c])
                    aligned_true[h][ch][t] = float(base_prices[k, c]) * float(g_true[k, j, c])

    return aligned_pred, aligned_true

def extend_x_dates(base_dates, aligned_pred: dict, horizons):
    """
    base_dates: (N,) 기준일 시계열
    aligned_pred: {h: {ch: 1D array}}
    -> 가장 긴 시리즈 길이에 맞춰 영업일로 x축 확장
    """
    x = _ensure_dtindex(base_dates)
    use_h = [int(h) for h in horizons if h in aligned_pred]
    if not use_h:
        raise ValueError(f"aligned_pred에 {list(horizons)} 키가 없습니다. keys={list(aligned_pred.keys())}")

    need_len = 0
    for h in use_h:
        for ch in CHANNELS:
            need_len = max(need_len, len(aligned_pred[h][ch]))

    if len(x) >= need_len:
        return x

    add = need_len - len(x)
    extra = pd.bdate_range(x[-1] + BDay(1), periods=add)
    return x.append(extra)

# ---------------- 테이블 생성 (멀티채널) ----------------
def build_target_aligned_table_multich(
    aligned_pred: Dict[int, Dict[str, np.ndarray]],
    aligned_true: Dict[int, Dict[str, np.ndarray]],
    x_dates,
    horizons
) -> pd.DataFrame:
    """
    열:
      date,
      true_high, true_low, true_close,
      pred_h{h}_high, pred_h{h}_low, pred_h{h}_close, ...
    """
    x_idx = _ensure_dtindex(x_dates)
    T = len(x_idx)
    df = pd.DataFrame({"date": x_idx})

    def _fit_len(arr, T):
        a = np.asarray(arr, dtype=float)
        if len(a) == T: return a
        if len(a) > T:  return a[:T]
        out = np.full(T, np.nan, dtype=float)
        out[:len(a)] = a
        return out

    use_h = [int(h) for h in horizons if h in aligned_pred]

    # 예측 열
    for h in use_h:
        for ch in CHANNELS:
            df[f"pred_h{h}_{ch}"] = _fit_len(aligned_pred[h][ch], T)

    # 실제 열(채널별): 작은 h부터 유효값 채움
    for ch in CHANNELS:
        series = np.full(T, np.nan, dtype=float)
        for h in sorted(use_h):
            arr = _fit_len(aligned_true[h][ch], T)
            m = np.isnan(series) & np.isfinite(arr)
            series[m] = arr[m]
        df[f"true_{ch}"] = series

    return df

# ---------------- (옵션) h=1 롤링 경로 ----------------
def build_rolling_h1_series(dfp: pd.DataFrame) -> pd.DataFrame:
    """
    dfp: build_target_aligned_table_multich 결과
    반환: date, roll_h1_high/low/close (h=1 예측을 날짜축에 붙인 경로)
    """
    out = pd.DataFrame({"date": dfp["date"]})
    for ch in CHANNELS:
        col = f"pred_h1_{ch}"
        out[f"roll_h1_{ch}"] = dfp[col].to_numpy(dtype=float)
    return out

# ---------------- 리포트 + 플롯 ----------------
def report_predictions(model, test_loader, device, cfg):
    """
    전제:
      - cfg.target_kind == 'close3_logr'
      - artifacts/scalers.pkl 존재, 구조: {'price': {'scaler': StandardScaler, 'cols': [...]}, ...}
      - test_loader batch dict: 'x_price','x_flow','x_fund','x_glob','y','base_date'
    """
    model.eval()
    horizons = list(cfg.horizons)
    target_kind = cfg.target_kind
    assert target_kind == "close3_logr", "현재 스크립트는 target_kind='close3_logr'만 지원합니다."
    assert len(horizons) >= 1, "horizons가 비어있습니다."

    # --- 스케일러 로드 & price 컬럼 인덱스 ---
    scalers_path = Path(cfg.scaler_dir) / "scalers.pkl"
    assert scalers_path.exists(), f"[ERR] scalers not found: {scalers_path}"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    price_pack = scalers.get("price", {})
    price_scaler = price_pack.get("scaler", None)
    assert price_scaler is not None and hasattr(price_scaler, "inverse_transform"), \
        "scalers['price']['scaler']가 없거나 inverse_transform을 지원하지 않습니다."
    price_idx_map = _price_indices_from_scaler_pack(price_pack)  # {'high': i, 'low': j, 'close': k}

    # --- 예측/정답/기준가/날짜 수집 ---
    Ps, Ys, base_prices_list, base_dates_list = [], [], [], []

    with torch.no_grad():
        for b in test_loader:
            yb = b["y"]                       # (B,H,3) — 로그수익률 라벨
            H = yb.size(1)
            # 모델은 (B,H,3) 로그수익률을 출력한다고 가정
            p = model(
                b["x_price"].to(device),
                b["x_flow"].to(device),
                b["x_fund"].to(device),
                b["x_glob"].to(device),
                H=H
            ).cpu().numpy()
            Ps.append(p)
            Ys.append(yb.numpy())

            # 기준가: 입력 price 마지막 시점(L-1)의 high/low/close를 역스케일
            x_last = b["x_price"][:, -1, :].cpu().numpy()  # (B, P_price)
            inv_all = price_scaler.inverse_transform(x_last)  # (B, P_price) 실가격 스케일
            h_idx = price_idx_map["high"]; l_idx = price_idx_map["low"]; c_idx = price_idx_map["close"]
            base_prices = np.stack(
                [inv_all[:, h_idx], inv_all[:, l_idx], inv_all[:, c_idx]], axis=1
            )  # (B,3)
            base_prices_list.append(base_prices)

            # 기준 날짜 (윈도우 끝 날짜)
            if "base_date" in b:
                base_dates_list.extend(b["base_date"])

    P = np.concatenate(Ps, axis=0)                    # (N, H, 3)  - 로그수익률
    Y = np.concatenate(Ys, axis=0)                    # (N, H, 3)  - 로그수익률
    base_prices = np.concatenate(base_prices_list, 0) # (N, 3)     - 절대가 기준
    assert P.shape == Y.shape and P.shape[-1] == 3, f"pred/true must match and last dim=3. got {P.shape} {Y.shape}"

    base_dates = pd.to_datetime(base_dates_list, errors="coerce", utc=True).tz_convert(None)
    if base_dates.isna().any() or len(base_dates) != len(P):
        raise RuntimeError("기준일 수집 실패 또는 길이 불일치. Dataset이 base_date를 반환하고, DataLoader(shuffle=False)인지 확인하세요.")

    # --- 정렬 배열 생성(멀티채널, 절대가 복원) ---
    aligned_pred, aligned_true = build_aligned_series_from_base_multich(
        preds=P, trues=Y, base_prices=base_prices, horizons=horizons, target_kind=target_kind
    )

    # --- x_dates 확장 ---
    x_dates = extend_x_dates(base_dates, aligned_pred, horizons)

    # --- 날짜별 테이블 생성 ---
    table = build_target_aligned_table_multich(
        aligned_pred=aligned_pred,
        aligned_true=aligned_true,
        x_dates=x_dates,
        horizons=horizons
    )
    # 종목명과 종목코드 컬럼 추가
    table.insert(1, "name", cfg.name)   # date 뒤에 삽입
    table.insert(2, "code", cfg.code)

    # --- 저장 ---
    out_dir = Path(cfg.predict_dir) / f"{cfg.end_date}"; out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"pred_{cfg.name}_{cfg.code}.csv"
    ordered_cols = (
        ["date", "name", "code"]
        + [f"true_{ch}" for ch in CHANNELS]
        + sum(([f"pred_h{h}_{ch}" for ch in CHANNELS] for h in horizons), [])
    )
    table[ordered_cols].to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # --- (옵션) h=1 롤링 경로 ---
    dfp = table.copy()
    dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
    x = dfp["date"].to_numpy()
    
    if cfg.PREDIC_PLOT and (1 in horizons):
        # --- 플롯: horizon별 × 채널별 ---
        for h in horizons:
            for ch in CHANNELS:
                pred_col = f"pred_h{h}_{ch}"
                true_col = f"true_{ch}"
                if pred_col not in dfp.columns:
                    continue

                y_true = dfp[true_col].astype(float).to_numpy()
                y_pred = dfp[pred_col].astype(float).to_numpy()

                m = np.isfinite(y_true) | np.isfinite(y_pred)
                if not m.any():
                    continue

                fig, ax = plt.subplots(figsize=(14,5))
                ax.plot(x[m], y_true[m], label=f"True {ch}", marker='o', linestyle='--', linewidth=2, alpha=0.9)
                ax.plot(x[m], y_pred[m], label=f"Pred +{h}d {ch}", marker='o', linestyle='-',  alpha=0.9)

                loc = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

                ax.set_title(f"{cfg.name}({cfg.code}) — {ch.upper()} | Target +{h}d")
                ax.set_xlabel("Date"); ax.set_ylabel("Price")
                ax.grid(True); ax.legend()
                plt.tight_layout()
                plt.show()

    # --- (옵션) h=1 롤링 경로 ---
    if cfg.PREDIC_ROLLING_H1_PLOT and (1 in horizons):
        # h=1 정렬 결과가 이미 절대가이므로 그 값을 날짜축에 그대로 붙여 시각화
        roll = pd.DataFrame({"date": dfp["date"]})
        for ch in CHANNELS:
            roll[f"roll_h1_{ch}"] = dfp[f"pred_h1_{ch}"].to_numpy(dtype=float)

        x2 = pd.to_datetime(roll["date"], errors="coerce").to_numpy()
        for ch in CHANNELS:
            y = roll[f"roll_h1_{ch}"].to_numpy(dtype=float)
            m = np.isfinite(y)
            if not m.any():
                continue
            fig, ax = plt.subplots(figsize=(14,4))
            ax.plot(x2[m], y[m], label=f"Rolling h=1 {ch}", marker='o', linestyle='-')
            loc = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            ax.set_title(f"{cfg.name}({cfg.code}) — Rolling h=1 {ch.upper()}")
            ax.set_xlabel("Date"); ax.set_ylabel("Price")
            ax.grid(True); ax.legend()
            plt.tight_layout()
            plt.show()

    

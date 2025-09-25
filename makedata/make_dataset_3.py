# make_dataset.py (revised for H×(1 or 3-channel) targets)
import os, pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from DSConfig_3 import DSConfig, SplitConfig, FeatureConfig

split = SplitConfig()
feature = FeatureConfig()  # 컬럼 정의 (price/flow/fund/global/date)

def build_target(df: pd.DataFrame, kind: str) -> Union[np.ndarray, np.ndarray]:
    """
    kind:
      - 'logr'        : log return of close (N,)
      - 'pct'         : pct change of close (N,)
      - 'close'       : close absolute (N,)
      - 'close3'      : absolute [high, low, close] (N, 3)
      - 'close3_logr' : per-channel log return of [high, low, close] (N, 3)
                        = [log(high).diff, log(low).diff, log(close).diff]
    """
    if kind == "logr":
        y = np.log(df["close"]).diff().fillna(0.0).to_numpy()

    elif kind == "pct":
        y = df["close"].pct_change().fillna(0.0).to_numpy()

    elif kind == "close":
        y = df["close"].to_numpy()

    elif kind == "close3":
        need = ["high", "low", "close"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"'{c}' column is required for target_kind='close3'.")
        y = df[need].to_numpy()  # (N,3)

    elif kind == "close3_logr":
        need = ["high", "low", "close"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"'{c}' column is required for target_kind='close3_logr'.")
        # 각 채널별 로그수익률 계산
        lh = np.log(df["high"]).diff().fillna(0.0).to_numpy()
        ll = np.log(df["low"]).diff().fillna(0.0).to_numpy()
        lc = np.log(df["close"]).diff().fillna(0.0).to_numpy()
        y = np.stack([lh, ll, lc], axis=1).astype(np.float32)  # (N,3)

    else:
        raise ValueError("target_kind must be one of "
                         "['logr','pct','close','close3','close3_logr']")

    return y.astype(np.float32, copy=False)

# ---------------- Split (time-ordered) ----------------
def time_split(df: pd.DataFrame, split: SplitConfig):
    n = len(df)
    i_tr = int(n * split.train_ratio)
    i_va = int(n * (split.train_ratio + split.val_ratio))
    tr = df.iloc[:i_tr].copy()
    va = df.iloc[i_tr:i_va].copy()
    te = df.iloc[i_va:].copy()
    return tr, va, te, i_tr, i_va

# ---------------- Scalers (fit on train only) ----------------
def fit_scalers(df: pd.DataFrame, feat_cfg: FeatureConfig):
    branches = [
        ("price", feat_cfg.price_cols),
        ("flow",  feat_cfg.flow_cols),
        ("fund",  feat_cfg.fund_cols),
        ("glob",  feat_cfg.global_cols),
    ]
    scalers = {}
    for branch, cols in branches:
        cols = [c for c in (cols or []) if c in df.columns]
        if not cols:
            continue
        scaler = StandardScaler()
        scaler.fit(df[cols].astype(float))
        scalers[branch] = {"scaler": scaler, "cols": cols}
    return scalers

def apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    out = df.copy()
    for branch, pack in scalers.items():
        cols   = pack["cols"]
        scaler = pack["scaler"]
        out[cols] = scaler.transform(out[cols].astype(float))
    return out

# ---------------- Make windows (inside each block only) ----------------
def make_windows_block(
    df_orig: pd.DataFrame,          # 원본(날짜/인덱스용)
    df_block: pd.DataFrame,         # 스케일링된 블록(tr/va/te)
    y_block: np.ndarray,            # (N,) or (N,3)
    lookback: int,
    horizons: Tuple[int, ...],
    feat_cfg: FeatureConfig,
    require_full_horizons: bool = True
):
    """
    반환:
      Xdate: (Nwin, L) datetime64[ns] 또는 None
      Xp   : (Nwin, L, Pp)
      Xf   : (Nwin, L, Pf)
      Xd   : (Nwin, L, Pd)
      Xg   : (Nwin, L, Pg)
      Y    : (Nwin, H)   if y_block.ndim == 1
             (Nwin, H,3) if y_block.ndim == 2 and y_block.shape[1]==3
    """
    Hs = list(horizons); Hmax = max(Hs)
    Xdate, Xp, Xf, Xd, Xg, Y = [], [], [], [], [], []

    # --- 컬럼 준비 ---
    pcols = [c for c in (feat_cfg.price_cols  or []) if c in df_block.columns]
    fcols = [c for c in (feat_cfg.flow_cols   or []) if c in df_block.columns]
    dcols = [c for c in (feat_cfg.fund_cols   or []) if c in df_block.columns]
    gcols = [c for c in (feat_cfg.global_cols or []) if c in df_block.columns]

    # --- 날짜 준비: 인덱스가 datetime이면 그대로, 아니면 지정된 date_col/cols에서 꺼냄 ---
    # 우선순위: df_orig.index 가 datetime → 그걸 사용
    if isinstance(df_orig.index, pd.DatetimeIndex):
        date_series = df_orig.loc[df_block.index].index.to_series()
    else:
        # date 컬럼명 추정
        date_cols = feat_cfg.date_cols if hasattr(feat_cfg, "date_cols") else []
        if isinstance(date_cols, str):
            date_cols = [date_cols]
        date_cols = [c for c in (date_cols or []) if c in df_orig.columns]
        if date_cols:
            date_series = df_orig.loc[df_block.index, date_cols[0]]
        else:
            date_series = None  # 날짜 없음

    # --- 넘파이 변환 (global이 비면 빈 2D로) ---
    def as_arr(df, cols):
        if not cols:
            # (n, 0) 빈 배열로 유지 → 윈도우 만들기 가능
            return np.empty((len(df), 0), dtype=np.float32)
        return df[cols].to_numpy(dtype=np.float32)

    price = as_arr(df_block, pcols)
    flow  = as_arr(df_block, fcols)
    fund  = as_arr(df_block, dcols)
    glob  = as_arr(df_block, gcols)

    n = len(df_block)
    t_end = (n - Hmax) if require_full_horizons else (n - 1)

    # 라벨 형식 파악
    y_block = np.asarray(y_block)
    y_is_3ch = (y_block.ndim == 2 and y_block.shape[1] == 3)

    for t in range(lookback - 1, t_end):
        s = t - lookback + 1

        # 날짜 윈도우
        if date_series is not None:
            win_dates = pd.to_datetime(date_series.iloc[s:t+1], errors="coerce")
            Xdate.append(win_dates.to_numpy(dtype="datetime64[ns]"))

        # 입력 윈도우
        Xp.append(price[s:t+1])
        Xf.append(flow[s:t+1])
        Xd.append(fund[s:t+1])
        Xg.append(glob[s:t+1])

        # 라벨 윈도우
        if require_full_horizons:
            if y_is_3ch:
                # (H,3)
                Y.append([y_block[t + h] for h in Hs])
            else:
                # (H,)
                Y.append([y_block[t + h] for h in Hs])
        else:
            # 끝까지 만들되 부족하면 NaN
            if y_is_3ch:
                row = []
                for h in Hs:
                    if (t + h) < n:
                        row.append(y_block[t + h])
                    else:
                        row.append([np.nan, np.nan, np.nan])
                Y.append(row)  # (H,3)
            else:
                row = []
                for h in Hs:
                    row.append(y_block[t + h] if (t + h) < n else np.nan)
                Y.append(row)  # (H,)

    Xdate_arr = None if not Xdate else np.asarray(Xdate, dtype="datetime64[ns]")
    Xp = np.asarray(Xp, np.float32)
    Xf = np.asarray(Xf, np.float32)
    Xd = np.asarray(Xd, np.float32)
    Xg = np.asarray(Xg, np.float32)
    Y  = np.asarray(Y,  np.float32)

    return (Xdate_arr, Xp, Xf, Xd, Xg, Y)

# ---------------- Main ----------------
def make_datasets(df: pd.DataFrame, cfg: DSConfig, LOAD_CSV_FILE=True):
    # 1) 원천 로드(선택)
    if LOAD_CSV_FILE:
        get_dir = Path(cfg.getdata_dir) / f"{cfg.end_date}"
        filepath = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.csv")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded data from: {filepath}")

    # 2) 타깃 전체 시계열
    y_full = build_target(df, cfg.target_kind)  # (N,) or (N,3)

    # 3) 시계열 분할
    tr, va, te, i_tr, i_va = time_split(df, split)
    if y_full.ndim == 1:
        y_tr, y_va, y_te = y_full[:i_tr], y_full[i_tr:i_va], y_full[i_va:]
    else:
        y_tr, y_va, y_te = y_full[:i_tr, :], y_full[i_tr:i_va, :], y_full[i_va:, :]

    # 4) 스케일러 (train에만 fit)
    scalers = fit_scalers(tr, feature)
    tr_s = apply_scalers(tr, scalers)
    va_s = apply_scalers(va, scalers)
    te_s = apply_scalers(te, scalers)

    # 5) 스케일러 저장
    scaler_dir = Path(cfg.scaler_dir); scaler_dir.mkdir(exist_ok=True, parents=True)
    scaler_path = os.path.join(scaler_dir, "scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    # 6) 윈도우 생성
    Xdate_tr, Xp_tr, Xf_tr, Xd_tr, Xg_tr, Y_tr = make_windows_block(
        df_orig=df, df_block=tr_s, y_block=y_tr,
        lookback=cfg.lookback, horizons=cfg.horizons, feat_cfg=feature,
        require_full_horizons=True
    )
    Xdate_va, Xp_va, Xf_va, Xd_va, Xg_va, Y_va = make_windows_block(
        df_orig=df, df_block=va_s, y_block=y_va,
        lookback=cfg.lookback, horizons=cfg.horizons, feat_cfg=feature,
        require_full_horizons=True
    )
    Xdate_te, Xp_te, Xf_te, Xd_te, Xg_te, Y_te = make_windows_block(
        df_orig=df, df_block=te_s, y_block=y_te,
        lookback=cfg.lookback, horizons=cfg.horizons, feat_cfg=feature,
        require_full_horizons=False  # 테스트는 끝까지
    )

    # 7) 메타 (FeatureConfig는 dict로 저장!)
    meta = {
        "feature_config": asdict(feature),  # 객체 대신 dict로!
        "lookback": cfg.lookback,
        "horizons": tuple(cfg.horizons),
        "target_kind": cfg.target_kind,
        "splits": {
            "train_ratio": split.train_ratio,
            "val_ratio": split.val_ratio,
            "test_ratio": split.test_ratio,
        },
        "lengths": {
            "train": int(len(Xp_tr)),
            "val":   int(len(Xp_va)),
            "test":  int(len(Xp_te)),
        },
    }

    # 8) payload
    payload = {
        # 학습
        "Xdate_tr": Xdate_tr, "Xp_tr": Xp_tr, "Xf_tr": Xf_tr, "Xd_tr": Xd_tr, "Xg_tr": Xg_tr, "Y_tr": Y_tr,
        # 검증
        "Xdate_va": Xdate_va, "Xp_va": Xp_va, "Xf_va": Xf_va, "Xd_va": Xd_va, "Xg_va": Xg_va, "Y_va": Y_va,
        # 테스트
        "Xdate_te": Xdate_te, "Xp_te": Xp_te, "Xf_te": Xf_te, "Xd_te": Xd_te, "Xg_te": Xg_te, "Y_te": Y_te,
        # 메타
        "meta": meta,
    }

    # 9) 저장
    out_dir = Path(cfg.dataset_dir) / f"{cfg.end_date}"; out_dir.mkdir(exist_ok=True, parents=True)
    dataset_path = os.path.join(out_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.pkl")
    with open(dataset_path, "wb") as f:
        pickle.dump(payload, f)

    print("Metadata:", meta)
    print(f"[saved] {dataset_path}")
    return payload

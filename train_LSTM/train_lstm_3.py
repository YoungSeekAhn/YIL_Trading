# train_lstm.py  (H-step × [high, low, close] 예측)
import math
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from DSConfig_3 import DSConfig
from train_LSTM.report_predic_3 import report_predictions  # <- 필요 시 이 함수도 3채널에 맞게 수정
import os

# ---------------- Dataset ----------------
class MIDataset(Dataset):
    def __init__(self, Xp, Xf, Xd, Xg, Y, base_dates=None):
        """
        Xp, Xf, Xd, Xg: (N, L, P*)
        Y: (N, H, 3)  # 마지막 축: [high, low, close]
        """
        self.Xp = np.asarray(Xp, dtype=np.float32)
        self.Xf = np.asarray(Xf, dtype=np.float32)
        self.Xd = np.asarray(Xd, dtype=np.float32)
        self.Xg = np.asarray(Xg, dtype=np.float32)
        self.Y  = np.asarray(Y,  dtype=np.float32)

        N = self.Y.shape[0]
        assert self.Xp.shape[0] == N and self.Xf.shape[0] == N and self.Xd.shape[0] == N and self.Xg.shape[0] == N, "N mismatch"
        assert self.Xp.shape[1] == self.Xf.shape[1] == self.Xd.shape[1] == self.Xg.shape[1], "lookback (L) mismatch"
        assert self.Y.ndim == 3 and self.Y.shape[-1] == 3, f"Y must be (N, H, 3). got {self.Y.shape}"

        # 날짜 처리
        self.base_dates = None
        if base_dates is not None:
            bd = np.asarray(base_dates)
            if bd.ndim == 2:
                bd = bd[:, -1]
            if bd.shape[0] != N:
                raise ValueError(f"base_dates length {bd.shape[0]} != N {N}")
            bd = pd.to_datetime(bd, errors="coerce", utc=True).tz_convert(None)
            if bd.isna().any():
                raise ValueError("base_dates contains NaT after parsing.")
            self.base_dates = [ts.isoformat() for ts in bd.to_pydatetime()]

    def __len__(self): return len(self.Y)

    def __getitem__(self, idx):
        item = {
            "x_price": torch.from_numpy(self.Xp[idx]),  # (L, Pp)
            "x_flow":  torch.from_numpy(self.Xf[idx]),  # (L, Pf)
            "x_fund":  torch.from_numpy(self.Xd[idx]),  # (L, Pd)
            "x_glob":  torch.from_numpy(self.Xg[idx]),  # (L, Pg)
            "y":       torch.from_numpy(self.Y[idx]),   # (H, 3)
        }
        if self.base_dates is not None:
            item["base_date"] = self.base_dates[idx]
        return item

# ---------------- Model ----------------
class BranchLSTM(nn.Module):
    def __init__(self, in_dim, hidden=96, layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.in_dim = int(in_dim)
        if self.in_dim > 0:
            self.lstm = nn.LSTM(
                input_size=self.in_dim,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=(dropout if layers > 1 else 0.0),
                bidirectional=bidirectional,
            )
            self.out_dim = hidden * (2 if bidirectional else 1)
        else:
            self.lstm = None
            self.out_dim = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        if self.in_dim == 0 or self.lstm is None:
            B = x.size(0)
            return x.new_zeros(B, self.out_dim)

        if x.size(-1) != self.in_dim:
            raise RuntimeError(
                f"[BranchLSTM] expected in_dim={self.in_dim}, got x[..., {x.size(-1)}]. "
                "전처리/모델 dims를 확인하세요."
            )

        _, (h, _) = self.lstm(x)         # h: (num_layers * d, B, hidden)
        if self.lstm.bidirectional:
            h_last = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden*2)
        else:
            h_last = h[-1]                              # (B, hidden)
        return h_last


class MultiBranchLSTM(nn.Module):
    def __init__(
        self,
        dims,                    # {'p': P_dim, 'f': F_dim, 'd': D_dim, 'g': G_dim}
        lstm_hidden=96,
        lstm_layers=1,
        lstm_dropout=0.2,
        bidirectional=False,
        mlp_hidden=128,
        out_dim=1,               # <- 최종 출력 차원(B, out_dim). 여기서는 H*3로 설정
        mlp_dropout=0.2,
    ):
        super().__init__()
        P = int(dims.get('p', 0))
        F = int(dims.get('f', 0))
        D = int(dims.get('d', 0))
        G = int(dims.get('g', 0))

        self.enc_p = BranchLSTM(P, lstm_hidden, lstm_layers, lstm_dropout, bidirectional)
        self.enc_f = BranchLSTM(F, lstm_hidden, lstm_layers, lstm_dropout, bidirectional)
        self.enc_d = BranchLSTM(D, lstm_hidden, lstm_layers, lstm_dropout, bidirectional)
        self.enc_g = BranchLSTM(G, lstm_hidden, lstm_layers, lstm_dropout, bidirectional)

        enc_out = self.enc_p.out_dim + self.enc_f.out_dim + self.enc_d.out_dim + self.enc_g.out_dim
        assert enc_out > 0, "At least one branch must have in_dim > 0."

        self.mlp = nn.Sequential(
            nn.Linear(enc_out, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, xp, xf, xd, xg, H=None):
        """
        반환: (B, H, 3)  # 3채널(high, low, close)
        """
        hp = self.enc_p(xp) if self.enc_p.out_dim > 0 else None
        hf = self.enc_f(xf) if self.enc_f.out_dim > 0 else None
        hd = self.enc_d(xd) if self.enc_d.out_dim > 0 else None
        hg = self.enc_g(xg) if self.enc_g.out_dim > 0 else None

        h_list = [t for t in (hp, hf, hd, hg) if t is not None]
        h = torch.cat(h_list, dim=-1)           # (B, enc_out)
        out = self.mlp(h)                       # (B, H*3)

        if H is None:
            # H를 명시하지 않으면 3으로 나누어 떨어진다고 가정
            assert out.size(-1) % 3 == 0, f"out_dim must be multiple of 3, got {out.size(-1)}"
            H = out.size(-1) // 3
        out = out.view(out.size(0), H, 3)       # (B, H, 3)
        return out

# ---------------- Metrics ----------------
def _rmse_mae(pred: np.ndarray, true: np.ndarray):
    rmse = math.sqrt(((pred - true) ** 2).mean())
    mae  = np.abs(pred - true).mean()
    return rmse, mae

# ---------------- Eval ----------------
def evaluate(model, device, criterion, loader):
    model.eval(); total=0; n=0; Ys=[]; Ps=[]
    with torch.no_grad():
        for b in loader:
            xp=b["x_price"].to(device); xf=b["x_flow"].to(device); xd=b["x_fund"].to(device); xg=b["x_glob"].to(device)
            y=b["y"].to(device)                 # (B, H, 3)
            H = y.size(1)
            p=model(xp,xf,xd,xg,H=H)            # (B, H, 3)

            loss=criterion(p, y)
            total+=loss.item()*y.size(0); n+=y.size(0)
            Ys.append(y.cpu()); Ps.append(p.cpu())

    Y=torch.cat(Ys,0).numpy()   # (N, H, 3)
    P=torch.cat(Ps,0).numpy()   # (N, H, 3)

    # 전체 채널 통합 지표
    rmse_all, mae_all = _rmse_mae(P, Y)

    # 채널별(high/low/close) 지표
    rmse_h, mae_h = _rmse_mae(P[...,0], Y[...,0])
    rmse_l, mae_l = _rmse_mae(P[...,1], Y[...,1])
    rmse_c, mae_c = _rmse_mae(P[...,2], Y[...,2])

    metrics = {
        "rmse": rmse_all, "mae": mae_all,
        "rmse_high": rmse_h, "mae_high": mae_h,
        "rmse_low":  rmse_l, "mae_low":  mae_l,
        "rmse_close":rmse_c, "mae_close":mae_c,
    }
    return total/max(1,n), metrics

# ---------------- Train ----------------
def train(model, cfg, device, criterion, train_loader, val_loader, num_epochs):
    grad_clip=1.0
    es_patience=6

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    best = float("inf"); bad = 0
    model_dir = Path(cfg.model_dir) /f"{cfg.end_date}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(model_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.pt")

    history={"train_loss":[],"val_loss":[],"val_rmse":[],"val_mae":[]}

    for epoch in range(1, num_epochs+1):
        model.train(); total=0; n=0
        print(f"\nEpoch {epoch}/{num_epochs}")

        pbar=tqdm(train_loader, desc="Training", leave=False)
        for b in pbar:
            xp=b["x_price"].to(device); xf=b["x_flow"].to(device); xd=b["x_fund"].to(device); xg=b["x_glob"].to(device)
            y=b["y"].to(device)             # (B,H,3)
            H=y.size(1)

            optimizer.zero_grad(set_to_none=True)
            p=model(xp,xf,xd,xg,H=H)        # (B,H,3)

            loss=criterion(p,y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total+=loss.item()*y.size(0); n+=y.size(0)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        tr_loss=total/max(1,n)
        va_loss, va_m = evaluate(model, device, criterion, val_loader)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_rmse"].append(va_m["rmse"])
        history["val_mae"].append(va_m["mae"])

        print(f"[Epoch {epoch:02d}] Train {tr_loss:.4f} | Val {va_loss:.4f} "
              f"| RMSE(all) {va_m['rmse']:.4f} | MAE(all) {va_m['mae']:.4f} "
              f"| RMSE[h/l/c] {va_m['rmse_high']:.4f}/{va_m['rmse_low']:.4f}/{va_m['rmse_close']:.4f} "
              f"| Best {best:.4f}")

        if va_m["rmse"] < best:
            best = va_m["rmse"]; bad = 0
            torch.save({"model": model.state_dict(), "target_kind": cfg.target_kind}, best_path)
        else:
            bad += 1
        if bad >= es_patience:
            print("Early stopping."); break

    return best_path, history

# ---------------- Top-level ----------------
def training_LSTM(payload, cfg, LOAD_DATASET_FILE = True, TRAIN_PLOT = True):
    if LOAD_DATASET_FILE:
        get_dir = Path(cfg.dataset_dir) / f"{cfg.end_date}"
        payload_path = os.path.join(get_dir, f"{cfg.name}({cfg.code})_{cfg.end_date}.pkl")
        payload = pd.read_pickle(payload_path)

    # 언팩
    Xdate_tr, Xp_tr, Xf_tr, Xd_tr, Xg_tr, Y_tr = \
        payload["Xdate_tr"], payload["Xp_tr"], payload["Xf_tr"], payload["Xd_tr"], payload["Xg_tr"], payload["Y_tr"]
    Xdate_va, Xp_va, Xf_va, Xd_va, Xg_va, Y_va = \
        payload["Xdate_va"], payload["Xp_va"], payload["Xf_va"], payload["Xd_va"], payload["Xg_va"], payload["Y_va"]
    Xdate_te, Xp_te, Xf_te, Xd_te, Xg_te, Y_te = \
        payload["Xdate_te"], payload["Xp_te"], payload["Xf_te"], payload["Xd_te"], payload["Xg_te"], payload["Y_te"]

    # Dataset
    ds_tr = MIDataset(Xp_tr, Xf_tr, Xd_tr, Xg_tr, Y_tr, Xdate_tr)
    ds_va = MIDataset(Xp_va, Xf_va, Xd_va, Xg_va, Y_va, Xdate_va)
    ds_te = MIDataset(Xp_te, Xf_te, Xd_te, Xg_te, Y_te, Xdate_te)

    print(f"Train dataset size: {len(ds_tr)} | Val dataset size: {len(ds_va)} | Test dataset size: {len(ds_te)}")

    # DataLoader
    train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    val_loader   = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # in_dim 파악
    sample = ds_tr[0]
    P_dim = sample["x_price"].shape[-1]
    F_dim = sample["x_flow"].shape[-1]
    D_dim = sample["x_fund"].shape[-1]
    G_dim = sample["x_glob"].shape[-1]
    H_dim = sample["y"].shape[0]      # H
    C_dim = sample["y"].shape[1]      # 3 (high, low, close)
    assert C_dim == 3, f"Y last dim must be 3(high,low,close). got {C_dim}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    dims = {'p': P_dim, 'f': F_dim, 'd': D_dim, 'g': G_dim}
    out_dim = H_dim * C_dim                    # H × 3
    model = MultiBranchLSTM(dims, out_dim=out_dim)
    model.to(device)

    best_model_path, history = train(model, cfg, device, criterion, train_loader, val_loader, num_epochs=30)
    print(f"Training completed. Best model saved to: {best_model_path}")

    # Best ckpt 로드 & 테스트
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_m = evaluate(model, device, criterion, test_loader)
    print(f"[TEST] loss {te_loss:.4f} | rmse(all) {te_m['rmse']:.4f} | mae(all) {te_m['mae']:.4f} "
          f"| RMSE[h/l/c] {te_m['rmse_high']:.4f}/{te_m['rmse_low']:.4f}/{te_m['rmse_close']:.4f}")

    # (선택) 학습 곡선
    if cfg.TRAIN_PLOT:
        fig, axes = plt.subplots(1,2, figsize=(12,5))
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].plot(history["val_loss"], label="Val")
        axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(history["val_rmse"], label="Val RMSE (all)")
        axes[1].plot(history["val_mae"], label="Val MAE (all)")
        axes[1].set_title("Val Errors"); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout(); plt.show()

    # 예측 리포트
    # report_predictions 함수가 기존 close 1채널 기준으로 되어 있다면,
    # (N,H,3) 입력을 받아 채널별 시각화/복원을 하도록 별도 수정이 필요합니다.
    report_predictions(model, test_loader, device, cfg)

    print("Training and evaluation completed successfully.")

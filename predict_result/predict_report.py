
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch LSTM evaluation report builder.
- Reads all CSVs in an input folder.
- Computes horizon-wise metrics (h1/h2/h3).
- Writes a per-file HTML report with cards + summary table (no detail tables).

Usage:
  python lstm_eval_batch.py --in ./data --out ./reports

CSV schema (wide format example):
  date,name,code,true_high,true_low,true_close,
  pred_h1_high,pred_h1_low,pred_h1_close,
  pred_h2_high,pred_h2_low,pred_h2_close,
  pred_h3_high,pred_h3_low,pred_h3_close
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

from DSConfig_3 import DSConfig, config
from makedata.dataset_functions import last_trading_day
from predict_result.predict_result import predict_result


STYLE = """
<style>
:root { --bg:#0b1020; --card:#141a33; --text:#e8ecff; --muted:#9aa3c7; --good:#22c55e; --ok:#eab308; --bad:#ef4444; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; padding: 24px; }
h1 { margin: 0 0 6px; font-size: 24px; }
p.sub { margin:0 0 18px; color: var(--muted); }
table { width:100%; border-collapse: collapse; }
table.summary { background: #0f152b; border-radius: 16px; overflow: hidden; margin-bottom: 18px; }
table.summary thead th { position: sticky; top: 0; background:#11183a; color:#c7d2fe; font-size:12px; padding:10px; border-bottom:1px solid #223; text-align:center; }
table.summary tbody td { padding: 10px; border-bottom:1px solid #1f2544; font-size:13px; text-align:center; }
.pill { padding: 2px 8px; border-radius: 999px; font-weight: 700; border:1px solid #334; }
.hor-h1 { color:#7dd3fc; border-color: rgba(125,211,252,.35); }
.hor-h2 { color:#a78bfa; border-color: rgba(167,139,250,.35); }
.hor-h3 { color:#f472b6; border-color: rgba(244,114,182,.35); }
.score { padding: 4px 10px; border-radius: 999px; font-weight: 800; }
.score.good { color: var(--good); border:1px solid rgba(34,197,94,.4); background:#0b1020; }
.score.ok { color: var(--ok); border:1px solid rgba(234,179,8,.35); background:#0b1020; }
.score.bad { color: var(--bad); border:1px solid rgba(239,68,68,.4); background:#0b1020; }
.section h2 { margin: 18px 0 8px; font-size: 18px; }
ul { margin: 8px 0 16px 18px; }
.kbd { background:#11183a; border:1px solid #223; border-radius:6px; padding:1px 6px; font-size:12px; }
.footer { color: var(--muted); font-size: 12px; margin-top: 12px; }
.muted { color: var(--muted); }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap: 12px; margin-bottom: 18px; }
.card { background: var(--card); border:1px solid #223; border-radius: 14px; padding: 12px; }
.card .k { color: var(--muted); font-size: 12px; }
.card .v { font-size: 18px; font-weight: 700; margin-top: 6px; }
</style>
"""

EXPLAIN = """
<div class="section">
  <h2>요약 지표 해석 가이드</h2>
  <ul>
    <li><b>종합점수 (0–100)</b>: 0.4×MAPE점수 + 0.3×방향정확도 + 0.2×밴드포착률 + 0.1×범위겹침률</li>
    <li>MAPE점수 = <span class="kbd">100 × (1 − min(MAPE, 50) / 50)</span> (MAPE 0%→100점, ≥50%→0점)</li>
    <li><b>MAPE/MAE/RMSE/Bias</b>: 모두 종가(close) 기준. 낮을수록 좋음(단, Bias는 0 근처가 바람직)</li>
    <li><b>Directional Accuracy</b>: 전일 종가 대비 방향(상승/하락) 일치율(%)</li>
    <li><b>true_close∈[pred_low,high]</b>, <b>범위 겹침률</b>: 예측 밴드의 포착/겹침 성능(%)</li>
  </ul>
  <p class="muted">※ 결측치는 자동 제외됩니다. 지표 가중치는 필요시 조정 가능합니다.</p>
</div>
"""

def fmt(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return f"{x:,.{d}f}"

def score_class(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    if v >= 75: return "good"
    if v >= 55: return "ok"
    return "bad"

def horizon_metrics(df, h):
    ph, pl, pc = f"pred_{h}_high", f"pred_{h}_low", f"pred_{h}_close"
    subset = df.dropna(subset=["true_high","true_low","true_close", ph, pl, pc]).copy()
    if subset.empty:
        return None
    subset["err_close"] = subset[pc] - subset["true_close"]
    subset["abs_err_close"] = subset["err_close"].abs()
    subset["mape_close"] = subset["abs_err_close"] / subset["true_close"] * 100.0
    mae = subset["abs_err_close"].mean()
    rmse = np.sqrt((subset["err_close"]**2).mean())
    mape = subset["mape_close"].mean()
    bias = subset["err_close"].mean()
    da_df = subset.dropna(subset=["prev_true_close"]).copy()
    if not da_df.empty:
        true_dir = np.sign(da_df["true_close"] - da_df["prev_true_close"])
        pred_dir = np.sign(da_df[pc] - da_df["prev_true_close"])
        dir_acc = (true_dir == pred_dir).mean() * 100.0
    else:
        dir_acc = np.nan
    within_close_band = ((subset["true_close"] >= subset[pl]) & (subset["true_close"] <= subset[ph])).mean() * 100.0
    overlap = ~((subset["true_high"] < subset[pl]) | (subset["true_low"] > subset[ph]))
    band_overlap_rate = overlap.mean() * 100.0
    mae_high = (subset[ph] - subset["true_high"]).abs().mean()
    mae_low  = (subset[pl] - subset["true_low"]).abs().mean()
    mape_score = 100.0 * (1.0 - min(max(mape, 0), 50) / 50.0)
    dir_score = 0.0 if pd.isna(dir_acc) else dir_acc
    band_score = 0.0 if pd.isna(within_close_band) else within_close_band
    overlap_score = 0.0 if pd.isna(band_overlap_rate) else band_overlap_rate
    composite = 0.4 * mape_score + 0.3 * dir_score + 0.2 * band_score + 0.1 * overlap_score
    return {
        "MAPE_close(%)": mape,
        "MAE_close": mae,
        "RMSE_close": rmse,
        "Bias_close": bias,
        "DirAcc_close(%)": dir_acc,
        "Close-in-Band(%)": within_close_band,
        "HL-Band-Overlap(%)": band_overlap_rate,
        "MAE_high": mae_high,
        "MAE_low": mae_low,
        "Composite(0-100)": composite
    }

def summary_table_html(summaries):
    head = """
    <table class="summary">
      <thead>
        <tr>
          <th>호라이즌</th>
          <th>종합점수<br>(0–100)</th>
          <th>MAPE<br>(close, %)</th>
          <th>MAE<br>(close)</th>
          <th>RMSE<br>(close)</th>
          <th>Bias<br>(close)</th>
          <th>Directional<br>Accuracy (%)</th>
          <th>true_close∈[pred_low,high]<br>(%)</th>
          <th>[true_low,high]∩[pred_low,high]<br>(%)</th>
          <th>MAE(high)</th>
          <th>MAE(low)</th>
        </tr>
      </thead>
      <tbody>
    """
    rows = []
    for h in ["h1","h2","h3"]:
        s = summaries.get(h)
        if s is None:
            rows.append(f"<tr><td><span class='pill hor-{h}'>{h}</span></td><td colspan='10' class='muted'>유효한 평가 데이터가 없습니다.</td></tr>")
            continue
        comp = s["Composite(0-100)"]
        rows.append(
            "<tr>" +
            f"<td><span class='pill hor-{h}'>{h}</span></td>" +
            f"<td><span class='score {score_class(comp)}'>{fmt(comp,1)}</span></td>" +
            f"<td>{fmt(s['MAPE_close(%)'],2)}</td>" +
            f"<td>{fmt(s['MAE_close'],2)}</td>" +
            f"<td>{fmt(s['RMSE_close'],2)}</td>" +
            f"<td>{fmt(s['Bias_close'],2)}</td>" +
            f"<td>{fmt(s['DirAcc_close(%)'],2)}</td>" +
            f"<td>{fmt(s['Close-in-Band(%)'],2)}</td>" +
            f"<td>{fmt(s['HL-Band-Overlap(%)'],2)}</td>" +
            f"<td>{fmt(s['MAE_high'],2)}</td>" +
            f"<td>{fmt(s['MAE_low'],2)}</td>" +
            "</tr>"
        )
    tail = "</tbody></table>"
    return head + "\n".join(rows) + tail

def build_report_for_file(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    name = df["name"].dropna().iloc[0]
    code = str(df["code"].dropna().iloc[0])
    for col in df.columns:
        if col in ["date","name","code"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["prev_true_close"] = df["true_close"].shift(1)
    summaries = {h: horizon_metrics(df, h) for h in ["h1","h2","h3"]}
    # pick best horizon
    best_h, best_score = None, -1
    for h, s in summaries.items():
        if s and s["Composite(0-100)"] > best_score:
            best_h, best_score = h, s["Composite(0-100)"]
    # mape score and dir acc
    if best_h:
        mape_score = 100.0 * (1.0 - min(max(summaries[best_h]["MAPE_close(%)"], 0), 50) / 50.0)
        dir_acc = summaries[best_h]["DirAcc_close(%)"]
    else:
        mape_score, dir_acc = np.nan, np.nan
    # header info
    valid = df.dropna(subset=["true_close"])
    if not valid.empty:
        start_date = valid["date"].min().strftime("%Y-%m-%d")
        end_date = valid["date"].max().strftime("%Y-%m-%d")
        sample_period = f"{start_date} ~ {end_date}"
        sample_count = len(valid)
    else:
        sample_period, sample_count = "데이터 없음", 0

    title = f"{name} ({code}) LSTM 예측 정확도 평가 리포트"
    cards = f"""
    <div class="cards">
      <div class="card"><div class="k">종합점수</div><div class="v">{fmt(best_score,1)}</div></div>
      <div class="card"><div class="k">선정 Horizon</div><div class="v">{best_h if best_h else ''}</div></div>
      <div class="card"><div class="k">MAPE점수</div><div class="v">{fmt(mape_score,1) if not np.isnan(mape_score) else ''}</div></div>
      <div class="card"><div class="k">방향정확도 (%)</div><div class="v">{fmt(dir_acc,1) if not np.isnan(dir_acc) else ''}</div></div>
    </div>
    """
    html = f"""<!doctype html>
<html lang="ko">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
{STYLE}
<body>
  <h1>{title}</h1>
  <p class="sub">샘플 데이터 기간: {sample_period} · 샘플수: {sample_count} 일 · 생성: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  {cards}
  <h2>요약 지표</h2>
  {summary_table_html(summaries)}
  {EXPLAIN}
  <div class="footer">© LSTM Eval — 자동 산출. 지표 정의/가중치는 필요 시 변경 가능합니다.</div>
</body>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"Report_pred_{name}_{code}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path

def predict_report(cfg):
 
    input_dir = Path(cfg.predict_result_dir) / f"{cfg.end_date}"
    
    output_dir = Path(cfg.predict_report_dir)
    output_dir = output_dir / f"Report_{cfg.end_date}"
    
    csv_files = sorted(input_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"[WARN] No CSV files found in {input_dir}")
        return

    for csv in csv_files:
        try:
            out_path = build_report_for_file(csv, output_dir)
            print(f"[OK] {csv.name} -> {out_path.name}")
        except Exception as e:
            print(f"[ERR] {csv.name}: {e}")

   # 공유 폴더로 복사(홈페이지 전시용)
    copy_path = Path(r"C:\Users\ganys\python_work\YIL_server\shared\reports\2_forecast")
    dest_path = copy_path / f"Report_{cfg.end_date}"
    try:
        dest_path.mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(output_dir, dest_path, dirs_exist_ok=True)
        print(f"Copied: {output_dir} -> {dest_path}")
        
    except Exception as e:
        print(f"[WARN] failed to copy HTML to shared reports: {e}")
        
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Input folder containing CSV files")
    ap.add_argument("--out", dest="outdir", required=True, help="Output folder for HTML reports")
    args = ap.parse_args()
    predict_report(config)
    
if __name__ == "__main__":

    main()

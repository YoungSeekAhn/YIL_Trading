# -*- coding: utf-8 -*-
"""
report_builder.py (wide-metrics version)
- 멀티타깃(h1/h2/h3 × high/low/close) 집계 CSV(여러 종목, wide 형태)를 읽어
  1) 종목별 1행 요약 CSV (wide)  → 입력 그대로 정리/정렬해 저장
  2) 상세 long CSV               → (종목 × horizon × target) 행으로 전개
  3) 종목별 HTML 리포트(표 + 시각화) → 집계지표 기반 바차트(시계열 미사용)
를 생성합니다.

입력 CSV 예시 헤더 (질문 제공 형식):
종목명,종목코드,
h1_high_RMSE,h1_high_MAE,h1_high_MAPE(%),h1_high_Bias,h1_high_count,
h1_low_RMSE,h1_low_MAE,h1_low_MAPE(%),h1_low_Bias,h1_low_count,
h1_close_RMSE,h1_close_MAE,h1_close_MAPE(%),h1_close_Bias,h1_close_count,h1_close_DirAcc(%),h1_close_Dir_count,
h2_high_RMSE,...,h2_close_DirAcc(%),h2_close_Dir_count,
h3_high_RMSE,...,h3_close_DirAcc(%),h3_close_Dir_count

의존: pandas, numpy, matplotlib
"""

import os, argparse
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from DSConfig_3 import DSConfig, config
from makedata.dataset_functions import last_trading_day
from predict_result.predict_result import predict_result


# ---- 고정 설정 ----
HORIZONS = [1, 2, 3]
TARGETS  = ["high", "low", "close"]

# ---------------------------
# 유틸
# ---------------------------
def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def ensure_code_str6(code) -> str:
    s = str(code).strip()
    return s.zfill(6) if s.isdigit() and len(s) < 6 else s

# ---------------------------
# wide → long 전개
# ---------------------------
def wide_row_to_long(row: pd.Series) -> List[Dict]:
    rows: List[Dict] = []
    name = row["종목명"]
    code = row["종목코드"]

    for h in HORIZONS:
        # close 방향성 지표 (horizon당 1쌍)
        dir_acc = row.get(f"h{h}_close_DirAcc(%)", np.nan)
        dir_cnt = row.get(f"h{h}_close_Dir_count", 0)

        for t in TARGETS:
            r = {
                "종목명": name,
                "종목코드": code,
                "horizon": f"h{h}",
                "target": t,
                "RMSE":  row.get(f"h{h}_{t}_RMSE",  np.nan),
                "MAE":   row.get(f"h{h}_{t}_MAE",   np.nan),
                "MAPE(%)": row.get(f"h{h}_{t}_MAPE(%)", np.nan),
                "Bias":  row.get(f"h{h}_{t}_Bias",  np.nan),
                "count": row.get(f"h{h}_{t}_count", 0),
                "DirAcc_close(%)": dir_acc if t == "close" else np.nan,
                "Dir_count":       dir_cnt if t == "close" else 0,
            }
            rows.append(r)
    return rows

# ---------------------------
# 리포트(HTML) 생성 (집계지표 기반)
# ---------------------------
def plot_close_mape_bar(row: pd.Series, out_png: str):
    xs = [f"h{h}" for h in HORIZONS]
    ys = [row.get(f"h{h}_close_MAPE(%)", np.nan) for h in HORIZONS]
    plt.figure(figsize=(5.2, 3.2))
    plt.bar(xs, ys)
    plt.title("Close MAPE(%) by Horizon")
    plt.xlabel("Horizon"); plt.ylabel("MAPE(%)")
    plt.tight_layout(); plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_png, dpi=140); plt.close()

def plot_close_diracc_bar(row: pd.Series, out_png: str):
    xs = [f"h{h}" for h in HORIZONS]
    ys = [row.get(f"h{h}_close_DirAcc(%)", np.nan) for h in HORIZONS]
    plt.figure(figsize=(5.2, 3.2))
    plt.bar(xs, ys)
    plt.title("Close Directional Accuracy(%) by Horizon")
    plt.xlabel("Horizon"); plt.ylabel("DirAcc(%)")
    plt.tight_layout(); plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_png, dpi=140); plt.close()

def plot_bias_bars(row: pd.Series, out_png: str):
    # horizon × target (3×3=9개) Bias를 한 그래프에 그룹화 막대
    labels = [f"h{h}" for h in HORIZONS]
    width  = 0.25
    x = np.arange(len(labels))
    series = []
    for i, t in enumerate(TARGETS):
        ys = [row.get(f"h{h}_{t}_Bias", np.nan) for h in HORIZONS]
        series.append((t, ys, x + (i-1)*width))

    plt.figure(figsize=(6.2, 3.4))
    for (t, ys, xpos) in series:
        plt.bar(xpos, ys, width, label=t)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Bias by Horizon × Target (pred - true)")
    plt.xticks(x, labels); plt.xlabel("Horizon"); plt.ylabel("Bias")
    plt.legend()
    plt.tight_layout(); plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_png, dpi=140); plt.close()

def write_symbol_report_html(row: pd.Series,
                             long_df: pd.DataFrame,
                             out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    name = row["종목명"]; code = row["종목코드"]

    # 그래프 생성
    img_mape   = os.path.join(out_dir, "bar_close_mape.png")
    img_diracc = os.path.join(out_dir, "bar_close_diracc.png")
    img_bias   = os.path.join(out_dir, "bar_bias_all.png")
    plot_close_mape_bar(row, img_mape)
    plot_close_diracc_bar(row, img_diracc)
    plot_bias_bars(row, img_bias)

    # 표 생성
    # wide: 해당 종목 1행만 pretty
    wide_cols = ["종목명", "종목코드"] + [c for c in row.index if c not in ["종목명","종목코드"]]
    wide_df = pd.DataFrame([row[wide_cols]])

    wide_html = wide_df.to_html(index=False,
                                float_format=lambda x: f"{x:.6g}" if isinstance(x, float) else x)
    long_html = long_df.to_html(index=False,
                                float_format=lambda x: f"{x:.6g}" if isinstance(x, float) else x)

    # HTML
    title = f"예측 리포트 - {name} ({code})"
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 22px; margin-bottom: 8px; }}
h2 {{ font-size: 18px; margin-top: 24px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th {{ background: #f7f7f7; }}
td:first-child, th:first-child {{ text-align: left; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="small">자동 생성: report_builder.py (wide-metrics)</div>

  <div class="card">
    <h2>요약(종목당 1행, wide)</h2>
    {wide_html}
  </div>

  <div class="card">
    <h2>집계 시각화</h2>
    <div><img src="bar_close_mape.png" alt="Close MAPE by Horizon"/></div>
    <div><img src="bar_close_diracc.png" alt="Close DirAcc by Horizon"/></div>
    <div><img src="bar_bias_all.png" alt="Bias by Horizon × Target"/></div>
  </div>

  <div class="card">
    <h2>상세(long)</h2>
    {long_html}
  </div>
</body>
</html>
"""
    out_html = os.path.join(out_dir, f"report_{name}_{code or 'NA'}.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html

# ---------------------------
# 일괄 처리 + 인덱스
# ---------------------------
def run_build(cfg, csv_path: str, out_dir: str,
              wide_csv: str = "symbol_metrics_wide.csv",
              long_csv: str = "symbol_metrics_long.csv"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 입력 wide CSV 로드
    df = pd.read_csv(csv_path)
    # 컬럼 존재 확인
    must_have = ["종목명", "종목코드"]
    for c in must_have:
        if c not in df.columns:
            raise ValueError(f"입력 CSV에 '{c}' 컬럼이 없습니다.")

    # 코드 보정
    df["종목코드"] = df["종목코드"].map(ensure_code_str6)

    # 숫자화
    for c in df.columns:
        if c not in ["종목명", "종목코드"]:
            df[c] = _safe_num(df[c])

    # 2) long 생성
    long_rows: List[Dict] = []
    for _, row in df.iterrows():
        long_rows.extend(wide_row_to_long(row))
    long_all = pd.DataFrame(long_rows)

    # 3) 종목별 리포트 생성
    links = []
    for _, row in df.iterrows():
        name = row["종목명"]; code = row["종목코드"]
        sym_dir = os.path.join(out_dir, f"{code}_{name}")
        os.makedirs(sym_dir, exist_ok=True)

        long_sub = long_all[(long_all["종목명"] == name) & (long_all["종목코드"] == code)].copy()
        html_path = write_symbol_report_html(row, long_sub, sym_dir)
        rel_path = os.path.relpath(html_path, out_dir)
        links.append((name, code, rel_path))

    # 4) CSV 저장
    # (wide는 정렬해서 깔끔하게 저장)
    sort_cols = ["종목명", "종목코드"]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    wide_path = os.path.join(out_dir, wide_csv)
    long_path = os.path.join(out_dir, long_csv)
    df_sorted.to_csv(wide_path, index=False, encoding="utf-8-sig")
    long_all.to_csv(long_path, index=False, encoding="utf-8-sig")

    # 5) 인덱스 HTML
    idx_html = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>예측 리포트 인덱스</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 22px; margin-bottom: 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f7f7f7; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</style>
</head>
<body>
  <h1>예측 리포트 인덱스</h1>
  <div class="small">자동 생성: report_builder.py (wide-metrics)</div>
  <table>
    <thead><tr><th>종목명</th><th>종목코드</th><th>리포트</th></tr></thead>
    <tbody>
      {}
    </tbody>
  </table>
  <p class="small">요약 CSV: <code>{}</code> / 상세 CSV: <code>{}</code></p>
</body>
</html>
"""
    rows_html = "\n".join(
        [f'<tr><td>{n}</td><td>{c}</td><td><a href="{p}">{os.path.basename(p)}</a></td></tr>'
         for (n, c, p) in links]
    )
    with open(os.path.join(out_dir, f"Report_index_{cfg.end_date}.html"), "w", encoding="utf-8") as f:
        f.write(idx_html.format(rows_html, os.path.basename(wide_path), os.path.basename(long_path)))

    print(f"[DONE] wide: {wide_path}")
    print(f"[DONE] long: {long_path}")
    print(f"[DONE] index: {os.path.join(out_dir, f"Report_index_{cfg.end_date}.html")}")


# ---------------------------
# CLI
# ---------------------------
def predict_report(cfg):

    predict_result(cfg)
    
    input_dir = Path(cfg.predict_report_dir)
    input_file = input_dir / f"Report_{cfg.end_date}" / f"Predict_result_{cfg.end_date}.csv"

    output_dir = Path(cfg.predict_report_dir)
    output_file = output_dir / f"Report_{cfg.end_date}"

    run_build(cfg, input_file, output_file)

def main():
    print("→ report_builder.py 시작")
    report_out(config)
    
if __name__ == "__main__":
    main()

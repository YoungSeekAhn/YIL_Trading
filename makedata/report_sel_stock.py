# -*- coding: utf-8 -*-
import os
import sys
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
from html import escape

# 프로젝트 루트 경로 주입
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DSConfig_3 import config
from makedata.dataset_functions import last_trading_day


def _fmt_code(val) -> str:
    # 한국 상장코드 6자리 0패딩
    try:
        s = str(val).strip()
        # 이미 6자리면 유지, 숫자로 보이면 0패딩
        if s.isdigit():
            return s.zfill(6)
        # '005930.0' 같은 경우 처리
        if s.replace('.', '', 1).isdigit():
            return str(int(float(s))).zfill(6)
        return s
    except Exception:
        return str(val)

def _pill_html(v) -> str:
    # 0/5/10 값에 따라 배지 색상
    try:
        iv = int(float(v))
    except Exception:
        return f'<span class="pill pill-na">–</span>'
    cls = "pill-0"
    if iv >= 10:
        cls = "pill-10"
    elif iv == 5:
        cls = "pill-5"
    elif iv <= 0:
        cls = "pill-0"
    return f'<span class="pill {cls}">{iv}</span>'

def _score_bar_html(val: float, vmax: float) -> str:
    try:
        v = float(val)
        mx = float(vmax) if vmax and vmax > 0 else 0.0
    except Exception:
        v, mx = 0.0, 0.0
    pct = 0 if mx <= 0 else max(0, min(100, int(round((v / mx) * 100))))
    return (
        f'<div class="score">'
        f'  <div class="bar" style="--w:{pct}%"></div>'
        f'  <span class="val">{v:,.1f}</span>'
        f'</div>'
    )

def report_sel_stock(cfg):
    # 입력 CSV 경로
    get_dir = Path(cfg.selout_dir)
    e_krx = cfg.end_date
    csv_path = get_dir / f"scored_{e_krx}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 점수 컬럼 자동 탐지
    score_col = next((c for c in df.columns if c.lower() == "score_1w"), None)
    if score_col is None:
        raise ValueError("Score_1w(또는 score_1w) 컬럼을 찾을 수 없습니다.")

    # 숫자 변환
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    # 코드 6자리 포맷
    if "Code" in df.columns:
        df["Code"] = df["Code"].apply(_fmt_code)

    # 상위 10개
    top10 = (
        df.sort_values(by=score_col, ascending=False)
          .head(10)
          .reset_index(drop=True)
    )

    # 표시 컬럼(존재하는 것만)
    base_cols = ["Name", "Code", score_col]
    sig_cols = ["rsi30","momentum_pos","macd_cross","ema5_over_ema20","ema20_over_ema60",
                "rs_plus","vol_120","vol_150","frg_own_1m_up","frg_3_pos","frg_5_pos",
                "ins_3_pos","ins_5_pos"]
    display_cols = [c for c in base_cols + sig_cols if c in top10.columns]
    top10 = top10[display_cols]

    # 헤더 라벨(보기 좋게)
    header_map = {
        "Name": "종목명", "Code": "코드", score_col: "Score(1w)",
        "rsi30": "RSI<30", "momentum_pos": "Momentum",
        "macd_cross": "MACD", "ema5_over_ema20": "EMA5>20",
        "ema20_over_ema60": "EMA20>60", "rs_plus": "RS+",
        "vol_120": "Vol120", "vol_150": "Vol150",
        "frg_own_1m_up": "외인지분↑1M", "frg_3_pos": "외인(3)",
        "frg_5_pos": "외인(5)", "ins_3_pos": "기관(3)", "ins_5_pos": "기관(5)",
    }

    # 최대 스코어(게이지 정규화)
    vmax = float(top10[score_col].max()) if not top10.empty else 0.0

    # 테이블 HTML 수동 빌드(셀별 스타일 적용 용이)
    thead_cells = ['<th class="col-rank">#</th>'] + [
        f'<th>{escape(header_map.get(c, c))}</th>' for c in top10.columns
    ]
    thead_html = "<tr>" + "".join(thead_cells) + "</tr>"

    rows_html = []
    for i, row in top10.iterrows():
        tds = [f'<td class="rank">{i+1}</td>']
        for col in top10.columns:
            if col == score_col:
                cell = _score_bar_html(row[col], vmax)
                tds.append(f'<td class="score-cell">{cell}</td>')
            elif col in ("Name", "Code"):
                tds.append(f'<td class="text">{escape(str(row[col]))}</td>')
            else:
                # 신호 컬럼은 배지 렌더링, 숫자 아니면 그대로
                try:
                    _ = float(row[col])
                    tds.append(f'<td class="sig">{_pill_html(row[col])}</td>')
                except Exception:
                    tds.append(f'<td class="text">{escape(str(row[col]))}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    tbody_html = "\n".join(rows_html)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 전체 HTML
    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top 10 by {escape(score_col)}</title>
<style>
:root {{
  --bg: #ffffff;
  --card: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --border: #e5e7eb;
  --row: #f9fafb;
  --accent: #2563eb;
  --pos: #16a34a;
  --mid: #f59e0b;
  --neg: #9ca3af;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0b1020;
    --card: #0f172a;
    --text: #e5e7eb;
    --muted: #a1a1aa;
    --border: #1f2937;
    --row: #0b1220;
    --accent: #3b82f6;
    --pos: #22c55e;
    --mid: #fbbf24;
    --neg: #6b7280;
  }}
}}
* {{ box-sizing: border-box; }}
html, body {{ height: 100%; }}
body {{
  margin: 0; padding: 24px;
  font-family: system-ui, Segoe UI, Arial, Apple SD Gothic Neo, Noto Sans KR, sans-serif;
  background: var(--bg); color: var(--text);
  letter-spacing: .1px; line-height: 1.45;
  -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
.header {{
  display:flex; align-items:baseline; gap:12px; margin-bottom: 12px;
}}
h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing:.2px; }}
.meta {{ color: var(--muted); font-size: 13px; }}
.legend {{ color: var(--muted); font-size: 13px; display:flex; gap:12px; margin: 8px 0 16px; flex-wrap: wrap; }}
.legend .pill {{ transform: translateY(-1px); }}
.card {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,.05);
}}
.table-wrap {{ overflow-x: auto; }}
table {{
  width: 100%; border-collapse: separate; border-spacing: 0;
  font-variant-numeric: tabular-nums;
}}
thead th {{
  position: sticky; top: 0; z-index: 1;
  background: var(--card); color: var(--muted);
  text-align: center; font-weight: 700; font-size: 13px;
  padding: 10px 8px; border-bottom: 1px solid var(--border);
  letter-spacing:.3px; text-transform: uppercase;
}}
tbody td {{
  padding: 10px 8px; border-bottom: 1px solid var(--border); text-align: center;
}}
tbody tr:hover {{ background: var(--row); }}
.col-rank, .rank {{ width: 48px; }}
.rank {{
  font-weight: 700; color: var(--muted);
}}
.text {{ text-align: left; white-space: nowrap; }}

.score-cell {{ min-width: 180px; }}
.score {{
  position: relative; height: 22px; display: flex; align-items: center; gap: 8px;
}}
.score .bar {{
  --w: 0%;
  position: relative; flex: 1 1 auto; height: 8px; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent) var(--w), rgba(127,127,127,.18) var(--w));
  outline: 1px solid rgba(127,127,127,.15);
}}
.score .val {{
  min-width: 64px; text-align: right; font-weight: 700; font-variant-numeric: tabular-nums;
}}

.pill {{
  display:inline-block; min-width: 28px; padding: 2px 8px;
  border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing:.3px;
  color: #fff; line-height: 1.4; border: 1px solid rgba(0,0,0,.08);
}}
.pill-10 {{ background: var(--pos); }}
.pill-5  {{ background: var(--mid); }}
.pill-0  {{ background: var(--neg); }}
.pill-na {{ background: var(--neg); opacity:.6; }}

.footer {{
  margin-top: 10px; color: var(--muted); font-size: 12px;
}}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>YIL – Top 10 종목 (정렬: {escape(score_col)} ↓)</h1>
      <div class="meta">생성시각: {escape(generated_at)}</div>
    </div>
    <div class="legend">
      <div><span class="pill pill-10">10</span> 긍정</div>
      <div><span class="pill pill-5">5</span> 중립</div>
      <div><span class="pill pill-0">0</span> 없음</div>
    </div>

    <div class="card table-wrap">
      <table>
        <thead>{thead_html}</thead>
        <tbody>
          {tbody_html}
        </tbody>
      </table>
    </div>

    <div class="footer">※ 본 화면은 자동 생성된 스코어 요약이며, 투자 권유가 아닙니다.</div>
  </div>
</body>
</html>
"""

    # 출력 경로: Sel_Report_{YYYYMMDD}.html
    out_dir = Path(cfg.selout_dir) / "Report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / f"Sel_Report_{cfg.end_date}.html"

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"[OK] HTML saved -> {out_html}")

    # 공유 폴더로 복사(홈페이지 전시용)
    copy_path = Path(r"C:\Users\ganys\python_work\YIL_server\shared\reports\1_select")
    try:
        copy_path.mkdir(parents=True, exist_ok=True)
        dst = copy_path / out_html.name
        shutil.copy(out_html, dst)
        print(f"[OK] HTML copied to -> {dst}")
    except Exception as e:
        print(f"[WARN] failed to copy HTML to shared reports: {e}")


if __name__ == "__main__":
    config.end_date = last_trading_day()
    report_sel_stock(config)

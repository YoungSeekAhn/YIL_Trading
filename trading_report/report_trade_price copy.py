# -*- coding: utf-8 -*-
"""
make_web_report_v4.py
- Trading_price_<end_date>.csv (v4 스키마 포함) 읽어 HTML 리포트 생성
- 신뢰도(confidence) 내림차순 정렬로 전시
- 신뢰도 0.5 이상 = 투자가능(행 강조 색상), 카드 영역도 0.5 이상만 표시
- 유효기간(valid_until)은 'YYYY-MM-DD'까지만 표시
- Score_1w: Trading_price에 없으면 scored_{end_date}.csv에서 병합 (종목코드→종목명 순)
- 파일 컬럼은 '원래 위치'로 하이퍼링크 (상대경로: <src_base>/<date>/<source_file>)
- 순수 HTML/CSS/JS (외부 의존 X)

사용 예:
    python make_web_report_v4.py  # DSConfig_3.config의 end_date 자동 계산 후 생성
"""
import html
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import sys

# 프로젝트 루트 경로를 sys.path에 추가 (trading_report/ 내부에서 실행해도 import 가능)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 외부 프로젝트 구성요소
from DSConfig_3 import config                    # 프로젝트 설정 객체
from makedata.dataset_functions import last_trading_day
from trading_report.make_trade_price import make_trade_price  # Trading_price_<date>.csv 생성 함수

# -------------------------
# 유틸
# -------------------------
def _fmt(x, nd=2):
    """숫자 포맷(소수 nd자리, 천단위 콤마). NaN/None은 빈문자."""
    try:
        if pd.isna(x):
            return ""
        if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer()):
            return f"{int(x):,}"
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)

def _fmt0(x):
    """현재가 전용: 정수 표기."""
    try:
        if pd.isna(x):
            return ""
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

def _safe(s):
    return html.escape("" if s is None else str(s))

def _class_by_rr(rr):
    try:
        if pd.isna(rr):
            return "tag-gray"
        rr = float(rr)
        if rr >= 2.5:
            return "tag-green"
        if rr >= 1.8:
            return "tag-blue"
        if rr >= 1.2:
            return "tag-amber"
        return "tag-red"
    except Exception:
        return "tag-gray"

def _class_by_conf(c):
    try:
        if pd.isna(c):
            return "bar-0"
        c = float(c)
        if c >= 0.8:
            return "bar-80"
        if c >= 0.6:
            return "bar-60"
        if c >= 0.4:
            return "bar-40"
        if c >= 0.2:
            return "bar-20"
        return "bar-0"
    except Exception:
        return "bar-0"

def _class_by_side(s):
    s = (s or "").upper()
    return "side-buy" if s == "BUY" else ("side-sell" if s == "SELL" else "side-unknown")

def _hlabel(h):
    return {"h1": "H1(1일)", "h2": "H2(2일)", "h3": "H3(3일)"}.get(str(h).lower(), str(h))

def _date_only(s: str) -> str:
    try:
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return ""
        return pd.to_datetime(str(s)).strftime('%Y-%m-%d')
    except Exception:
        return str(s)[:10]

def _is_investable(c) -> bool:
    try:
        return float(c) >= 0.5
    except Exception:
        return False

def _norm_code(s: pd.Series) -> pd.Series:
    """종목코드 표준화: 문자열화, .0 제거, 하이픈 제거, 숫자형이면 6자리 zero-fill."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    s = s.str.replace("-", "", regex=False)
    return s.apply(lambda x: x.zfill(6) if x.isdigit() else x)

# -------------------------
# 컬럼 정규화/Score 병합
# -------------------------
COL_ALIASES = {
    "h1_MAPE(%)": ["h1_MAPE(%)", "h1_close_MAPE(%)"],
    "h2_MAPE(%)": ["h2_MAPE(%)", "h2_close_MAPE(%)"],
    "h3_MAPE(%)": ["h3_MAPE(%)", "h3_close_MAPE(%)"],
    "h1_DirAcc(%)": ["h1_DirAcc(%)", "h1_close_DirAcc(%)"],
    "h2_DirAcc(%)": ["h2_DirAcc(%)", "h2_close_DirAcc(%)"],
    "h3_DirAcc(%)": ["h3_DirAcc(%)", "h3_close_DirAcc(%)"],
}

def _attach_score_1w(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    df에 Score_1w가 없으면 scored_{end}.csv에서 붙임.
    우선키: 종목코드 -> 예비키: 종목명
    """
    if "Score_1w" in df.columns:
        return df

    score_path = Path(cfg.selout_dir) / f"scored_{cfg.end_date}.csv"
    if not score_path.exists():
        print(f"[WARN] score file not found: {score_path}")
        df["Score_1w"] = pd.NA
        return df

    sc = pd.read_csv(score_path, dtype=str).rename(columns={"Name": "종목명", "Code": "종목코드"})
    if "종목코드" in sc.columns:
        sc["종목코드"] = _norm_code(sc["종목코드"])
    if "종목코드" in df.columns:
        df["종목코드"] = _norm_code(df["종목코드"])

    if "Score_1w" in sc.columns:
        sc["Score_1w"] = pd.to_numeric(sc["Score_1w"], errors="coerce")
    else:
        sc["Score_1w"] = pd.NA

    merged = df.copy()
    done = False
    if "종목코드" in merged.columns and "종목코드" in sc.columns:
        merged = merged.merge(sc[["종목코드", "Score_1w"]].drop_duplicates("종목코드"),
                              on="종목코드", how="left")
        done = True
    if not done and ("종목명" in merged.columns and "종목명" in sc.columns):
        merged = merged.merge(sc[["종목명", "Score_1w"]].drop_duplicates("종목명"),
                              on="종목명", how="left")
        done = True
    if not done:
        print("[WARN] cannot attach Score_1w (no common key).")
        merged["Score_1w"] = pd.NA

    return merged

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """테이블/카드 전시에 필요한 기본 컬럼 보장 + 대체명 매핑."""
    df = df.copy()
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    for out_col, cands in COL_ALIASES.items():
        if out_col not in df.columns:
            for c in cands:
                if c in df.columns:
                    df[out_col] = df[c]
                    break
    need = [
        "종목명","종목코드","권장호라이즌","last_close",
        "매수가(entry)","익절가(tp)","손절가(sl)","RR",
        "Score_1w",
        "ord_qty","side","confidence","holding_days","valid_until","source_file",
    ]
    for c in need:
        if c not in df.columns:
            df[c] = ""
    return df

# -------------------------
# HTML 생성
# -------------------------
def build_html(df: pd.DataFrame, df_cards: pd.DataFrame, with_cards: bool, date_token: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(df)
    avg_rr = df["RR"].dropna().mean() if "RR" in df.columns else float("nan")
    avg_conf = df["confidence"].dropna().mean() if "confidence" in df.columns else float("nan")
    buy_cnt = (df["side"].astype(str).str.upper() == "BUY").sum() if "side" in df.columns else 0
    sell_cnt = (df["side"].astype(str).str.upper() == "SELL").sum() if "side" in df.columns else 0
    hz_counts = df["권장호라이즌"].value_counts(dropna=False).to_dict() if "권장호라이즌" in df.columns else {}

    # 테이블 행
    rows_html = []
    for _, r in df.iterrows():
        rr_cls   = _class_by_rr(r.get("RR"))
        conf_cls = _class_by_conf(r.get("confidence"))
        side_cls = _class_by_side(r.get("side"))
        hz_raw   = str(r.get("권장호라이즌", "")).lower()
        hz       = _hlabel(hz_raw)
        row_cls  = "row-invest" if _is_investable(r.get("confidence")) else "row-non"

        rows_html.append(f"""
<tr class="{row_cls}" data-hz="{_safe(hz_raw)}">
  <td>{_safe(r.get("종목명"))}</td>
  <td>{_safe(r.get("종목코드"))}</td>
  <td><span class="hz">{_safe(hz)}</span></td>
  <td class="num">{_fmt0(r.get("last_close"))}</td>
  <td class="num">{_fmt(r.get("매수가(entry)"))}</td>
  <td class="num">{_fmt(r.get("익절가(tp)"))}</td>
  <td class="num">{_fmt(r.get("손절가(sl)"))}</td>
  <td class="num"><span class="tag {rr_cls}">{_fmt(r.get("RR"),2)}</span></td>
  <td class="num">{_fmt(r.get("Score_1w"),2)}</td>
  <td class="num">{(int(r.get("ord_qty")) if (not pd.isna(r.get("ord_qty")) and str(r.get('ord_qty')).strip()!='') else "")}</td>
  <td><span class="side {side_cls}">{_safe(r.get("side"))}</span></td>
  <td>
    <div class="bar"><div class="fill {conf_cls}" style="width:{int(100*float(r.get('confidence') or 0))}%;"></div></div>
    <div class="bar-text">{_fmt(r.get("confidence"),3)}</div>
  </td>
  <td class="num">{(int(r.get("holding_days")) if (not pd.isna(r.get("holding_days")) and str(r.get('holding_days')).strip()!='') else "")}</td>
  <td class="mono">{_date_only(r.get("valid_until"))}</td>
</tr>
""")

    # 카드 섹션: 신뢰도 0.5 이상만 표시
    cards_html = []
    if with_cards:
        for _, r in df_cards.iterrows():
            rr_cls   = _class_by_rr(r.get("RR"))
            conf_cls = _class_by_conf(r.get("confidence"))
            hz       = _hlabel(r.get("권장호라이즌", ""))
            cards_html.append(f"""
<div class="card">
  <div class="card-hd">
    <div class="title">{_safe(r.get("종목명"))} <span class="muted">({_safe(r.get("종목코드"))})</span></div>
    <div class="pill">{_safe(hz)}</div>
  </div>
  <div class="grid">
    <div><div class="k">현재가</div><div class="v">{_fmt0(r.get("last_close"))}</div></div>
    <div><div class="k">매수가</div><div class="v">{_fmt(r.get("매수가(entry)"))}</div></div>
    <div><div class="k">익절가</div><div class="v">{_fmt(r.get("익절가(tp)"))}</div></div>
    <div><div class="k">손절가</div><div class="v">{_fmt(r.get("손절가(sl)"))}</div></div>
    <div><div class="k">RR</div><div class="v"><span class="tag {rr_cls}">{_fmt(r.get("RR"),2)}</span></div></div>
    <div><div class="k">Score_1w</div><div class="v">{_fmt(r.get("Score_1w"),2)}</div></div>
    <div><div class="k">수량</div><div class="v">{(int(r.get("ord_qty")) if (not pd.isna(r.get("ord_qty")) and str(r.get('ord_qty')).strip()!='') else "")}</div></div>
    <div><div class="k">보유일</div><div class="v">{(int(r.get("holding_days")) if (not pd.isna(r.get("holding_days")) and str(r.get('holding_days')).strip()!='') else "")}</div></div>
    <div><div class="k">유효기간</div><div class="v mono">{_date_only(r.get("valid_until"))}</div></div>
  </div>
  <div class="conf">
    <div class="conf-label">신뢰도</div>
    <div class="bar big"><div class="fill {conf_cls}" style="width:{int(100*float(r.get('confidence') or 0))}%;"></div></div>
    <div class="bar-text">{_fmt(r.get("confidence"),3)}</div>
  </div>
</div>
""")

    # 호라이즌 분포 뱃지
    hz_badges = " ".join([f'<span class="badge">{_hlabel(k)} <b>{v}</b></span>' for k, v in hz_counts.items()])

    # HTML 템플릿 (Score_1w 열 추가됨)
    tpl = """<!doctype html>
<html lang=\"ko\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>YIL-LABs 자동매매 리포트</title>
<style>
:root { --bg:#0b0f15; --card:#101722; --muted:#8aa0b2; --txt:#e7eef6; --green:#1fb981; --blue:#3b82f6; --amber:#f59e0b; --red:#ef4444; --gray:#6b7280; --line:#1f2a37; --mono:#cbd5e1; }
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Noto Sans KR,Arial,sans-serif}
.container{max-width:1280px;margin:24px auto;padding:0 16px}
h1{font-size:22px;margin:8px 0 4px} .muted{color:var(--muted)}
.kpi{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:16px 0}
.kpi .box{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.kpi .k{font-size:12px;color:var(--muted)} .kpi .v{font-size:20px;font-weight:700}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.badge{background:#0f172a;border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:var(--muted)} .badge b{color:var(--txt)}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
input[type=\"search\"],select{padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:#0b1220;color:var(--txt)}
.tablewrap{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:1040px}
th,td{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left;white-space:nowrap}
th{background:#0d1420;position:sticky;top:0;z-index:1;cursor:pointer}
td.num{text-align:right} .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mono)}
.tag{padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.tag-green{background:rgba(31,185,129,.15);color:#8ff1cc;border-color:rgba(31,185,129,.35)}
.tag-blue{background:rgba(59,130,246,.15);color:#b7d4ff;border-color:rgba(59,130,246,.35)}
.tag-amber{background:rgba(245,158,11,.15);color:#ffe2a7;border-color:rgba(245,158,11,.35)}
.tag-red{background:rgba(239,68,68,.15);color:#ffb3b3;border-color:rgba(239,68,68,.35)}
.tag-gray{background:#111827;color:#9ca3af}
.hz{font-weight:600}
.side{padding:2px 8px;border-radius:8px}
.side-buy{background:rgba(31,185,129,.15);color:#8ff1cc}
.side-sell{background:rgba(239,68,68,.15);color:#ffb3b3}
.side-unknown{background:#111827;color:#9ca3af}
.bar{height:6px;background:#0b1220;border:1px solid var(--line);border-radius:999px;overflow:hidden;min-width:120px}
.bar.big{height:10px;min-width:220px}
.fill{height:100%}
.fill.bar-80{background:linear-gradient(90deg,#1fb981,#3b82f6)}
.fill.bar-60{background:#3b82f6}
.fill.bar-40{background:#38bdf8}
.fill.bar-20{background:#f59e0b}
.fill.bar-0{background:#374151}
.bar-text{font-size:11px;color:var(--muted);margin-top:4px}
.cardgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.card-hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.card .title{font-weight:700} .pill{background:#0d1420;border:1px solid var(--line);border-radius:999px;padding:4px 10px;color:var(--muted)}
.card .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.card .k{font-size:11px;color:var(--muted)} .card .v{font-weight:600}
.alink{color:#93c5fd;text-decoration:none} .alink:hover{text-decoration:underline}
/* 투자 가능/불가 행 강조 */
tr.row-invest{background:rgba(31,185,129,.08)}
tr.row-non{background:rgba(239,68,68,.06)}
/* 가독성: 특정 열 최소폭 */
th:nth-child(1){min-width:120px} th:nth-child(2){min-width:90px} th:nth-child(10){min-width:90px} th:nth-child(13){min-width:120px}
footer{color:var(--muted);font-size:12px;margin:12px 0}
@media (max-width:900px){.kpi{grid-template-columns:repeat(2,1fr)}.card .grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class=\"container\">
  <h1>자동매매 리포트 <span class=\"muted\">(생성: __NOW__)</span></h1>

  <div class=\"kpi\">
    <div class=\"box\"><div class=\"k\">종목 수</div><div class=\"v\">__N__</div></div>
    <div class=\"box\"><div class=\"k\">평균 RR</div><div class=\"v\">__AVG_RR__</div></div>
    <div class=\"box\"><div class=\"k\">평균 신뢰도</div><div class=\"v\">__AVG_CONF__</div></div>
    <div class=\"box\"><div class=\"k\">포지션</div><div class=\"v\">BUY __BUY__ / SELL __SELL__</div></div>
    <div class=\"box\"><div class=\"k\">호라이즌 분포</div><div class=\"badges\">__HZ_BADGES__</div></div>
  </div>

  <div class=\"controls\">
    <input id=\"q\" type=\"search\" placeholder=\"종목명/코드 검색…\" oninput=\"filterTable()\">
    <select id=\"hz\" onchange=\"filterTable()\">
      <option value=\"\">호라이즌 전체</option>
      <option value=\"h1\">H1(1일)</option>
      <option value=\"h2\">H2(2일)</option>
      <option value=\"h3\">H3(3일)</option>
    </select>
    <select id=\"side\" onchange=\"filterTable()\">
      <option value=\"\">포지션 전체</option>
      <option value=\"BUY\">BUY</option>
      <option value=\"SELL\">SELL</option>
    </select>
    <label>최소 신뢰도 <input id=\"minc\" type=\"range\" min=\"0\" max=\"100\" value=\"50\" oninput=\"showC(this.value);filterTable()\"></label>
    <span id=\"minc_val\" class=\"muted\">50%</span>
    <label>최소 RR <input id=\"minrr\" type=\"number\" step=\"0.1\" value=\"0\" style=\"width:80px\" oninput=\"filterTable()\"></label>
  </div>

  <div class=\"tablewrap\">
    <table id=\"t\">
      <thead>
        <tr>
          <th onclick=\"sortTable(0)\">종목명</th>
          <th onclick=\"sortTable(1)\">코드</th>
          <th onclick=\"sortTable(2)\">권장</th>
          <th onclick=\"sortTable(3)\">현재가</th>
          <th onclick=\"sortTable(4)\">매수가</th>
          <th onclick=\"sortTable(5)\">익절가</th>
          <th onclick=\"sortTable(6)\">손절가</th>
          <th onclick=\"sortTable(7)\">RR</th>
          <th onclick=\"sortTable(8)\">Score_1w</th>
          <th onclick=\"sortTable(9)\">수량</th>
          <th onclick=\"sortTable(10)\">포지션</th>
          <th onclick=\"sortTable(11)\">신뢰도</th>
          <th onclick=\"sortTable(12)\">보유일</th>
          <th onclick=\"sortTable(13)\">유효기간</th>
        </tr>
      </thead>
      <tbody>
        __ROWS__
      </tbody>
    </table>
  </div>

  __CARDS__

<footer>
※ RR = (익절폭 ÷ 손실폭). 값이 클수록 위험 대비 기대수익이 큽니다.<br>
※ 신뢰도(confidence)는 예측 품질을 0~1로 정규화한 지표로, 0.5 이상을 투자 가능 구간으로 표시합니다.<br>
※ Score_1w는 최근 1주 종합 스코어(테크니컬/수급 등)로, 클수록 매수 관심도가 높음을 의미합니다.<br>
※ 본 리포트는 융일 LABs 자동생성 결과이며 투자 결정의 최종 책임은 투자자 본인에게 있습니다.
</footer>

</div>

<script>
// 정렬
let sortDir = 1, lastCol = -1;
function sortTable(col){
  const t = document.getElementById('t');
  const tbody = t.tBodies[0];
  const rows = Array.from(tbody.rows);
  sortDir = (lastCol===col)? -sortDir : 1; lastCol = col;
  // 숫자열 인덱스: 3(현재가),4,5,6,7(RR),8(Score),9(수량),11(신뢰도),12(보유일)
  const numCols = [3,4,5,6,7,8,9,11,12];
  rows.sort((a,b)=>{
    let av = a.cells[col].innerText.replace(/[ ,]/g,'');
    let bv = b.cells[col].innerText.replace(/[ ,]/g,'');
    if(numCols.includes(col)){ av=parseFloat(av)||-Infinity; bv=parseFloat(bv)||-Infinity; }
    return (av>bv?1:av<bv?-1:0)*sortDir;
  });
  rows.forEach(r=>tbody.appendChild(r));
}

// 필터
function showC(v){ document.getElementById('minc_val').innerText = v + '%'; }
function filterTable(){
  const q    = document.getElementById('q').value.toLowerCase();
  const hz   = document.getElementById('hz').value.toLowerCase();
  const side = document.getElementById('side').value.toLowerCase();
  const minc = parseInt(document.getElementById('minc').value||'0',10);
  const minrr= parseFloat(document.getElementById('minrr').value||'0');
  const t = document.getElementById('t');
  const rows = Array.from(t.tBodies[0].rows);
  rows.forEach(tr=>{
    const name = tr.cells[0].innerText.toLowerCase();
    const code = tr.cells[1].innerText.toLowerCase();
    const hzc  = (tr.getAttribute('data-hz')||'').toLowerCase();
    const sidec= tr.cells[10].innerText.toLowerCase();  // 포지션
    const conf = parseFloat(tr.cells[11].innerText)||0;  // 신뢰도
    const rr   = parseFloat(tr.cells[7].innerText)||0;   // RR
    let ok = true;
    if(q && !(name.includes(q) || code.includes(q))) ok=false;
    if(hz && hzc!==hz) ok=false;
    if(side && sidec!==side) ok=false;
    if(conf*100 < minc) ok=false;
    if(rr < minrr) ok=false;
    tr.style.display = ok? '' : 'none';
  });
}
</script>
</body>
</html>
"""

    cards_block = ("<h2 style='margin-top:18px'>종목 카드</h2><div class='cardgrid'>" + "".join(cards_html) + "</div>") if with_cards else ""
    html_txt = (tpl
        .replace("__NOW__", _safe(now))
        .replace("__N__", str(n))
        .replace("__AVG_RR__", _fmt(avg_rr, 2))
        .replace("__AVG_CONF__", _fmt(avg_conf, 3))
        .replace("__BUY__", _fmt(buy_cnt, 0))
        .replace("__SELL__", _fmt(sell_cnt, 0))
        .replace("__HZ_BADGES__", hz_badges)
        .replace("__ROWS__", "".join(rows_html))
        .replace("__CARDS__", cards_block)
    )
    return html_txt

# -------------------------
# 리포트 메인
# -------------------------
def report_trade_price(cfg):
    # 1) 트레이딩 가격 CSV 생성 (이미 Score_1w 포함될 수 있음)
    make_trade_price(cfg)

    # 2) CSV 읽기
    input_csv = Path(cfg.price_report_dir) / f"Report_{cfg.end_date}" / f"Trading_price_{cfg.end_date}.csv"
    df = pd.read_csv(input_csv, dtype={"종목코드": str})

    # 3) Score_1w 없으면 scored 파일에서 병합
    df = _attach_score_1w(df, cfg)

    # 4) 전시용 컬럼 보장 및 대체명 매핑
    df = _normalize_columns(df)

    # 5) 카드용 (신뢰도 0.5 이상만)
    if 'confidence' in df.columns:
        df_cards = df[pd.to_numeric(df['confidence'], errors='coerce') >= 0.5].copy()
    else:
        df_cards = df.copy()

    # 6) HTML 생성
    html_text = build_html(df, df_cards, with_cards=True, date_token=cfg.end_date)

    out_html = Path(cfg.price_report_dir) / f"Report_{cfg.end_date}" / f"Trading_Report{cfg.end_date}.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_text)
    print(f"[OK] HTML saved -> {out_html}")

    # 7) 공유 폴더로 복사(홈페이지 전시용)
    copy_path = Path(r"C:\Users\ganys\python_work\YIL_server\shared\reports\3_price")
    try:
        copy_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_html, copy_path / out_html.name)
        print(f"[OK] HTML copied to -> {copy_path / out_html.name}")
    except Exception as e:
        print(f"[WARN] failed to copy HTML to shared reports: {e}")

# -------------------------
# 실행 스크립트
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # 마지막 거래일을 end_date로 세팅하여 생성
    end_date = last_trading_day()
    config.end_date = end_date
    report_trade_price(config)

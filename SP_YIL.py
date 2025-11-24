
# sp_yil_batch_training program

import os
import sys
import traceback
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from pykrx import stock

from makedata.sel_stock_ndays import sel_stock
from DSConfig_3 import config
from makedata.dataset_functions import last_trading_day
from makedata.get_dataset_2 import get_dataset
from makedata.make_dataset_3 import make_datasets
from train_LSTM.train_lstm_3 import training_LSTM
from predict_result.predict_report import predict_report
from predict_result.predict_result import predict_result
from trading_report.report_trade_price import report_trade_price
from makedata.report_sel_stock import report_sel_stock

# =========================
# 사용자 설정
# =========================
TOP_N = 20
DURATION_DAYS = 365 * 2          # 데이터 기간
SAVE_CSV_FILE = True             # get_dataset 수집 CSV 저장 여부
FORCE_REBUILD_DATASET = False    # True면 기존 .pkl 있어도 새로 make_datasets
PLOT_ROLLING = False             # (사용 중이면) 롤링 차트 그릴지 여부

TEST_MODE = False               # True면 end_date를 20250924로 고정 (테스트용)

# =========================
# 유틸
# =========================
def _ensure_code(code_or_none: Optional[str], name: str) -> str:
    """CSV에 Code가 비어 있으면 종목명으로 pykrx에서 코드 조회."""
    if code_or_none and str(code_or_none).strip():
        code = str(code_or_none).strip()
        # 6자리 zero-pad 보정
        if code.isdigit() and len(code) < 6:
            code = code.zfill(6)
        return code

    tickers = stock.get_market_ticker_list(market="ALL")
    mapping = {stock.get_market_ticker_name(t): t for t in tickers}
    if name not in mapping:
        raise ValueError(f"종목 '{name}'의 코드를 찾을 수 없습니다.")
    return mapping[name]

def _compute_dates():
    end_date = last_trading_day()  # "YYYYMMDD"
    
    if TEST_MODE:
        end_date = "20250924"
    
    start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=config.duration)
    start_date = start_dt.strftime("%Y%m%d")
    
    config.start_date = start_date
    config.end_date = end_date
    return start_date, end_date

def _paths_for(config) -> tuple[Path, Path]:
    """종목별 수집 CSV, 학습 pkl 경로 반환"""
    get_dir = Path(config.getdata_dir) / f"{config.end_date}"
    dataset_dir = Path(config.dataset_dir) / f"{config.end_date}"
    get_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csvpath = get_dir / f"{config.name}({config.code})_{config.end_date}.csv"
    datapath = dataset_dir / f"{config.name}({config.code})_{config.end_date}.pkl"
    return csvpath, datapath

def run_for_one(name: str, code: Optional[str]):
    # ----- 코드/기간/설정 세팅 -----
    code = _ensure_code(code, name)
    start_date = config.start_date
    end_date = config.end_date

    # DSConfig 갱신 (주의: 전역 config 객체를 루프마다 업데이트)
    config.name = name
    config.code = code

    # # 필요 시 추가 옵션도 동기화
    # if hasattr(config, "plot_rolling"):
    #     config.plot_rolling = PLOT_ROLLING

    csvpath, datapath = _paths_for(config)

    print(f"\n=== [{name} ({code})] {start_date}~{end_date} 시작 ===")
    print(f"- CSV: {csvpath}")
    print(f"- PKL: {datapath}")

    # ----- 원천 데이터 CSV 생성/존재 확인 -----
    merged_df = None
    if not csvpath.exists():
        print("→ 수집 CSV 없음 → get_dataset 실행")
        merged_df = get_dataset(config, SAVE_CSV_FILE=SAVE_CSV_FILE)
        print(f"   [OK] 수집 CSV 생성 완료: {csvpath}")
    else:
        print("→ 수집 CSV 이미 존재 (재수집 생략). 필요 시 get_dataset로 갱신 가능")

    # ----- 학습 데이터셋 생성/로드 -----
    payload = None
    if FORCE_REBUILD_DATASET or (not datapath.exists()):
        print("→ 학습 pkl 없음 또는 재생성 강제 → make_datasets(LOAD_CSV_FILE=False)")
        # merged_df가 None이라면 make_datasets 내부에서 csvpath를 사용하도록 구현되어 있다면 OK.
        # 아니라면 다음 줄에서 CSV를 읽어 merged_df를 만들어 전달하세요.
        # merged_df = pd.read_csv(csvpath, index_col=0, parse_dates=True)
        payload = make_datasets(merged_df, config, LOAD_CSV_FILE=False)
        print(f"   [OK] 학습 pkl 생성 완료: {datapath}")
            # ----- 학습 -----
        print("→ training_LSTM 시작")
        training_LSTM(payload, config)
        print(f"[DONE] {name} ({code})")
    else:
        print("→ 학습 pkl 이미 존재 → make_datasets(LOAD_CSV_FILE=True)로 로드 시도")
        # NOTE: make_datasets가 LOAD_CSV_FILE=True일 때 pkl을 로드해 payload를 반환하도록
        # 구현되어 있어야 합니다. 그렇지 않다면 직접 pickle 로딩 코드를 추가하세요.
        payload = make_datasets(None, config, LOAD_CSV_FILE=True)
        print("   [OK] 기존 pkl 로드 완료")
        print(" [SKIP] 학습 pkl이 이미 존재하므로 재생성 및 학습 생략")
    

# =========================
# 메인 실행
# =========================
def main():
    # 선정 CSV 로드
    #  CSV는 sel_stock_ndays.py 결과물이어야 함
    start_date, end_date= _compute_dates()
    
    config.start_date = start_date
    config.end_date = end_date
    
    sel_dir = Path(config.selout_dir)
    SEL_CSV_PATH = os.path.join(sel_dir, f"scored_{end_date}.csv")
    
    if not os.path.exists(SEL_CSV_PATH):
        #raise FileNotFoundError(f"선정 CSV를 찾을 수 없습니다: {SEL_CSV_PATH}")
        sel_stock(config)  # 선정 CSV 생성 시도
        report_sel_stock(config)  # 선정 리포트 생성 시도

    sel = pd.read_csv(SEL_CSV_PATH, dtype={"Code": str})
    # 안전한 정렬 (Score_1w 내림차순)
    if "Score_1w" not in sel.columns:
        raise ValueError("CSV에 'Score_1w' 컬럼이 없습니다.")
    sel = sel.sort_values(by="Score_1w", ascending=False).reset_index(drop=True)

    # 상위 N 추출 (Name, Code만 필요)
    cols_needed = ["Name", "Code"]
    for c in cols_needed:
        if c not in sel.columns:
            raise ValueError(f"CSV에 '{c}' 컬럼이 없습니다.")
    top = sel.loc[:TOP_N - 1, cols_needed].copy()

    print("===============================================")
    print(f"sel_stock_ndays 상위 {TOP_N} 종목 학습 파이프라인 시작")
    print("===============================================")
    results = []
    for i, row in top.iterrows():
        name = str(row["Name"]).strip()
        code = None if pd.isna(row["Code"]) else str(row["Code"]).strip()
        try:
            run_for_one(name, code)
            results.append((name, code, "OK", ""))
        except Exception as e:
            err = "".join(traceback.format_exception_only(type(e), e)).strip()
            print(f"[ERROR] {name} 실패: {err}")
            # 스택 요약 로그 (필요 시 상세 traceback.print_exc())
            traceback.print_exc()
            results.append((name, code, "FAIL", err))

    # 요약
    print("\n=========== 배치 요약 ===========")
    ok = [r for r in results if r[2] == "OK"]
    fail = [r for r in results if r[2] == "FAIL"]
    for tag, arr in [("성공", ok), ("실패", fail)]:
        print(f"\n[{tag} {len(arr)}]")
        for (n, c, _, msg) in arr:
            print(f"- {n} ({c}) {('→ ' + msg) if msg else ''}")

    # LSTM 학습/예측 결과 평가 및 리포트
    # predict_result(config) 함수 포함됨
    print("\n=========== 예측 결과 평가 및 리포트 ===========")
    predict_report(config)
    
    # 거래 가격 리포트
    # make_trade_price() 함수 포함됨
    print("\n=========== 거래 가격 리포트 ===========")
    report_trade_price(config)
    

import schedule
import time
import argparse
from datetime import datetime

def run_job():
    print(f"[RUN] main() 실행 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    main()

if __name__ == "__main__":
    # 명령줄 인자 파싱
    ap = argparse.ArgumentParser(description="자동매매 스케줄러")
    ap.add_argument("--start", default="22:00", help="시작 시간 (HH:MM 형식, 기본값 22:00)")
    args = ap.parse_args()

    # 입력 검증
    try:
        hour, minute = map(int, args.start.split(":"))
        assert 0 <= hour < 24 and 0 <= minute < 60
    except Exception:
        print("❌ 시작 시간 형식 오류. 예시: --start 21:30")
        exit(1)

    print(f"🕒 스케줄러 시작: 매일 {args.start}에 main() 실행")

    # 스케줄 등록
    schedule.every().day.at(args.start).do(run_job)

    # 무한 루프 (1분 단위로 확인)
    while True:
        schedule.run_pending()
        time.sleep(60)

# py -m PyInstaller --onefile SP_YIL.py

# python -m PyInstaller ^
#  --noconfirm ^
#  --clean ^
#  --noconsole ^
#  --icon=icon.ico ^
#  --add-data "config/.env;config/" ^
#  src/auto_trader.py
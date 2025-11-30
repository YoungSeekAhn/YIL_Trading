# DSConfig.py
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test_ratio는 1 - train - val
    test_ratio: float = 0.15
    shuffle: bool = False    # 시계열이라 보통 False 권장
    
@dataclass
class DSConfig:
    name: str = ""       # 종목 이름
    code: str = ""       # 종목 코드 (자동설정)
    
    duration: int = 365 * 2  # 데이터 기간 (일수)
    start_date: str = ""  # 자동 결정 (예: "20220101")
    end_date: str = ""    # 자동 결정 (예: "20231231")
    
    # 데이터 관련 설정
    lookback: int = 30     # 과거 데이터 길이 (일수)
    horizons: List[int] = (1, 2, 3)  # 예측 시점 (1일, 2일, 3일 뒤)
    target_kind: str = "close3_logr"   # 타겟 종류: "logr", "pct", "close", "close3",
    
    ## 분할 설정    
    #split: SplitConfig = field(default_factory=SplitConfig)
    batch_size: int = 32  # 배치 크기
    # 저장 경로
    selout_dir: str = "./_selec_out"  # 선택된 출력 결과 저장 디렉토리
    getdata_dir: str = "./_getdata" # CSV 수집 데이터 저장 디렉토리
    
    dataset_dir: str = "./_train/_datasets"  # 데이터셋 저장 디렉토리
    model_dir: str = "./_train/_models"  # 모델 저장 디렉토리
    scaler_dir: str = "./_train/_scalers"  # 스케일러 저장 디렉토리
         
    predict_result_dir: str = "./_predict_result"  # 출력 결과 저장 디렉토리
    predict_report_dir: str = "./_predict_report"  # 예측 결과 저장 디렉토리
    price_report_dir: str = "./_price_report"  # 분석 결과 저장 디렉토리
    
    env_dir: str = "./kis_trade"  # .env 파일 경로 (자동매매용)


    #test_getdata_dir: str = "./TR_LSTM3/_csvdata/삼성전자(005930)_20250909.csv"

    ## Display Option
    GETDATA_PLOT_ROLLING: bool = False  # get_dataset()에서 롤링 차트 그릴지 여부
    TRAIN_PLOT: bool = False  # training_LSTM()에서 학습 곡선 그릴지 여부
    PREDIC_PLOT: bool = False  # report_predic_3.py에서 롤링 차트 그릴지 여부
    PREDIC_ROLLING_H1_PLOT: bool = False  # report_predic_3.py에서 h=1 롤링 차트 그릴지 여부
    
config = DSConfig()

@dataclass
class FeatureConfig:
    date_cols: List[str] = ("date")
    price_cols: List[str] = ("open", "high", "low", "close", "volume", "chg_pct")
    flow_cols: List[str] = ("inst_sum", "inst_ext", "retail", "foreign")
    fund_cols: List[str] = ("per", "pbr", "div")
    global_cols: List[str] = ("KOSPI", "KOSDAQ", "SN500", "NASDAQ", "eps", "dps")

feature = FeatureConfig()

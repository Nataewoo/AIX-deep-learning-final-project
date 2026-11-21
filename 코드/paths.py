"""Shared repository paths; numerical/model settings stay in the source modules."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "데이터"
DATA_DIR = DATA_ROOT / "전처리"
SPLIT_DIR = DATA_ROOT / "분할"
RAW_DIR = DATA_ROOT / "원본"
CV_DIR = REPO / "실험결과" / "교차검증"
TEST_DIR = REPO / "실험결과" / "테스트"
TREE_DIR = REPO / "실험결과" / "추가트리비교"
FINAL_MODELS = REPO / "모델" / "최종"
COMPARISON_MODELS = REPO / "모델" / "비교"
FIGURE_DIR = REPO / "그림"
ENV_DIR = REPO / "실행환경"


def model_file(task, role):
    directory = FINAL_MODELS if role == "selected" else COMPARISON_MODELS
    return directory / f"model_{task}_{role}.joblib"

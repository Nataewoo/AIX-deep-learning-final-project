"""Shared repository paths; numerical/model settings stay in the source modules."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "데이터(Dataset)"
DATA_DIR = DATA_ROOT / "전처리"
SPLIT_DIR = DATA_ROOT / "분할"
RAW_DIR = DATA_ROOT / "원본"
RESULTS_DIR = REPO / "실험결과(Experiment Results)"
CV_DIR = RESULTS_DIR / "교차검증"
TEST_DIR = RESULTS_DIR / "테스트"
TREE_DIR = RESULTS_DIR / "추가트리비교"
MODEL_DIR = REPO / "모델(Model)"
FINAL_MODELS = MODEL_DIR / "최종"
COMPARISON_MODELS = MODEL_DIR / "비교"
FIGURE_DIR = REPO / "그림(Figure)"
ENV_DIR = REPO / "실행환경(Runtime Environment)"
DEMO_DIR = REPO / "예측시연(Prediction Demo)"


def model_file(task, role):
    directory = FINAL_MODELS if role == "selected" else COMPARISON_MODELS
    return directory / f"model_{task}_{role}.joblib"

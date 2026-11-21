# 실행 환경

읽기만 할 때는 설치 없이 [README 본문](../README.md)를 연다. 아래 명령은 **저장소 루트**에서 실행한다.

## 모델 학습 환경

원래 학습은 Python **3.11.9**에서 수행됐다. 패키지 고정값은 [requirements.txt](requirements.txt), 실제 저장된 버전은 [versions.json](versions.json)에 있다. 저장 기록의 PyTorch는 `2.7.1+cu128`이다. CUDA 구성은 사용 환경에 따라 달라질 수 있다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 실행환경/requirements.txt
```

## 저장 모델로 예측만 실행

이번 확인에 사용한 Python **3.10.11**과 [시연용 버전](requirements-demo.txt)으로 실행할 수 있다. 학습용 PyTorch 설치는 필요 없다.

```powershell
py -3.10 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r 실행환경/requirements-demo.txt
.\.venv-demo\Scripts\python.exe 코드/predict.py --input 예측시연/입력예시.json --output 예측시연/예측결과.json
```

시연은 NumPy로 저장된 MLP를 계산하므로 PyTorch 학습을 실행하지 않는다.

분할·전처리 자료를 다시 생성할 때도 위 시연 환경을 사용할 수 있다.

## 전처리·학습·검증 순서

```powershell
.\.venv\Scripts\python.exe 코드/make_splits.py --output .recheck/splits
.\.venv\Scripts\python.exe 코드/prepare.py
.\.venv\Scripts\python.exe 코드/evaluate.py
.\.venv\Scripts\python.exe 코드/verify.py
.\.venv\Scripts\python.exe 코드/compare_trees.py
```

분할 코드는 저장된 참가자 배정을 재사용한다. 예시에서는 `.recheck/splits`로 출력하여 동봉 자료를 보존한다. 분할을 실제 갱신하려면 `--output 데이터/분할`을 지정한다. 전처리는 `데이터/전처리/`, 모델 학습은 `모델/`, `실험결과/`, 학습 곡선은 `그림/`을 갱신한다. `evaluate.py`는 교차검증·최종 선택·재학습·별도 테스트 평가를 함께 수행한다.

## 저장 결과 재검산·오차 분석

기존 분석의 Python 3.12 환경 고정값은 [requirements-analysis.txt](requirements-analysis.txt)다. 원래 학습 환경과 구분한다.

```powershell
py -3.12 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r 실행환경/requirements-analysis.txt
.\.venv-analysis\Scripts\python.exe 코드/analyze_results.py --output .recheck
```

위 명령은 원본 CSV 재구성, 저장 예측·평가 지표·MLP 가중치 검산과 오차 그래프를 `.recheck/실험결과/상세분석/`, `.recheck/그림/`에 생성한다. `--output`을 생략하면 저장소의 상세 분석과 그림을 갱신한다. 모델을 다시 학습하지 않는다.

현재 확인한 Python 3.10 환경에서는 아래 명령으로 같은 검산을 실행했다.

```powershell
.\.venv-demo\Scripts\python.exe -m pip install -r 실행환경/requirements-verified.txt
.\.venv-demo\Scripts\python.exe 코드/analyze_results.py --output .recheck
```

현재 저장소에서 실제 확인한 환경과 실행 범위는 [재검증 기록](../실험결과/재검증/실행확인.json)에 별도로 남긴다. 원래 학습 전체를 다시 실행했다는 의미는 아니다.

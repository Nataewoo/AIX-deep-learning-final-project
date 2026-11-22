# 코드

저장소 루트에서 실행한다. [환경 설치·실행 순서](../실행환경/README.md)

| 파일 | 하는 일 |
| --- | --- |
| `prepare.py` | 원본에서 입력·정답과 과거 이력 생성 |
| `make_splits.py` | 기존 참가자 그룹으로 5회 학습·검증 분할 재생성 |
| `baseline.py` | 단순 기준선 |
| `mlp.py` | 딥러닝 모델 구조와 학습 |
| `ridge_regression.py` · `random_forest.py` · `gradient_boosting.py` · `xgboost_regressor.py` | 비교 알고리즘 |
| `evaluate.py` | 교차검증·설정 선택·전체 재학습·별도 평가 |
| `compare_trees.py` | Random Forest·XGBoost 추가 비교 |
| `verify.py` | 과거 이력과 저장 모델의 PyTorch 예측 검증 |
| `analyze_results.py` | NumPy 예측 검산·지표 재계산·오차 분석·그래프 생성 |
| `make_blog_figures.py` | 기존 자료·모델 설정·검증 결과로 본문 설명 그림 4개 생성 |
| `predict.py` | 학습 없이 저장 모델로 예측 시연 |
| `paths.py` | 저장소의 공통 파일 경로 |

전처리·학습·분석 코드는 제공된 소스를 기준으로 파일 경로를 조정했다. `make_splits.py`, `predict.py`, `paths.py`는 저장소에서 실행하기 위해 추가했다. 모델의 입력 계산식, 탐색 설정과 기존 결과는 유지했다. 원래 파일과의 대응은 [자료 출처 기록](../실행환경/자료_출처.json)에 있다.

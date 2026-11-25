# 데이터 처리와 모델 평가 재실행

Python 3.11이 설치된 Windows에서 저장소를 내려받은 뒤 `코드(Code)/Sleep_Model_Reproduction` 폴더에서 PowerShell을 연다. 아래 명령은 패키지를 설치할 때 인터넷 연결을 사용한다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare.py "../../데이터(Dataset)" --repo-layout
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe compare_trees.py
```

`--repo-layout`은 저장소의 [원본 자료](/데이터%28Dataset%29/원본)와 [분할 정보](/데이터%28Dataset%29/분할)를 사용한다. 저장소의 `test_melidos.csv`는 재현 코드 안에서 `validation_melidos.csv`로 저장하며, 현재 실행에서는 모델 선택을 마친 뒤 진행하는 해외 별도 평가에 사용한다. 원래 형식의 자료를 가지고 있다면 `prepare.py source_data`로도 실행할 수 있다.

이 재현 코드는 이 폴더 아래 `data/`, `results/`, `tree_comparison/`에 새 결과를 기록한다. 블로그에서 인용한 결과는 기존 [실험 결과 폴더](/실험결과%28Experiment%20Results%29)에서 확인할 수 있다. 자료 출처는 [Technical Blog 8장](/Technical%20Blog.md#8-참고자료-related-work)에 있다.

`prepare.py`는 팀 패키지의 기존 참가자 분할과 원본 레코드 번호를 사용한다. 원본에서 분 단위 입면 지연, 수면 구간과 수면의 질을 재구성하고, 참가자별로 아침 날짜순으로 정렬하여 각 밤보다 앞선 기록에서만 입력을 만든다. MeLiDos는 침대에 들어간 시각부터 잠을 청한 시각까지의 시간과 설문상 입면 지연을 합산한다. 국내 원본은 시계 시각의 차이를 자정을 고려해 계산하고, 배포된 시간 단위 값과 일치하는지 확인한다. 밤중 각성은 수면 구간 길이에 포함된다.

동일 참가자·수면 날짜 중복, 분할 간 참가자 중복, 지원되지 않는 시간 형식, 팀 패키지와의 주요 값 불일치는 실행 중 확인한다. 날짜가 없거나 필수 시각이 빠진 경우 관련 값은 결측으로 유지한다. MeLiDos에서 시각이 모두 있는 경우 잠들기까지의 시간은 침대에 들어간 시각부터 기상 시각까지의 시간보다 작아야 한다. 매우 긴 값을 길다는 이유만으로 삭제하지 않는다.

| 모델 | 입력 | 정답 |
|---|---|---|
| A | 직전 최대 3개 유효 입면 지연 중앙값과 기록 수, 직전 기록의 수면 구간, 직전 최대 3개 유효 수면 구간 중앙값과 기록 수 | `bed_to_sleep_min` |
| B | A와 동일한 다섯 변수 및 `sleep_interval_min` | `sleep_quality` (1–5점) |

`prev_sleep_interval_min`은 바로 앞선 날짜의 **기록**을 사용한다. 누락된 날짜가 있으면 직전 기록이 어제와 다를 수 있다. 바로 앞 기록의 구간이 비어 있으면 결측을 유지한다. 세 건 이력은 각 지표별로 유효한 기록을 센다. 이력이 없는 경우 중앙값은 결측이며 건수는 0이다. 취침 시각, 요일, 참가자 ID, A의 현재 밤 예측값은 기본 입력에서 제외한다.

`evaluate.py`는 MeLiDos Train의 참가자 단위 5-fold에서 모든 모델과 설정을 비교한 뒤 설정을 `selection.json`에 저장한다. 결측 대체와 Ridge·MLP 표준화는 각 fold의 학습 부분에서 적합한다. 학습 epoch는 교차검증에서 60/160 중 선택한다. MLP는 세 시드 모델의 예측 평균을 사용한다. 최종 평가에서는 전체 Train으로 다시 학습한 모델을 사용한다.

모델 구현 파일은 `baseline.py`, `ridge_regression.py`, `gradient_boosting.py`, `mlp.py`, `random_forest.py`, `xgboost_regressor.py`이다. Random Forest와 XGBoost의 추가 비교는 `compare_trees.py`에서 실행한다. XGBoost 비교에는 `xgboost-cpu==3.2.0`을 사용했다.

| 파일 | 내용 |
|---|---|
| `data/*.csv` | 분할별로 정리한 변수와 정답 |
| `data/summary.json` | 각 분할의 참가자·유효 행·결측 건수 |
| `results/cv_summary.csv`, `cv_by_fold.csv` | 설정별 통합 교차검증 점수와 fold별 점수 |
| `results/cv_groups.csv`, `selection.json` | 참가자별 fold와 최종 모델 설정 |
| `results/holdout_metrics.csv`, `holdout_predictions.csv` | 최종 평가 지표와 행별 예측 |
| `results/paired_intervals.csv` | 참가자 단위 재표집에 따른 MAE 차이의 95% 구간 |
| `results/model_*.joblib` | 선택 모델, 비교용 MLP, 입력 변환 및 가중치 |
| `results/learning_curves.png`, `learning_curves.csv` | 선택된 MLP 설정의 첫 fold 학습 추이 |

국내 마지막 7일 평가는 각 평가 밤 이전에 사용자가 실제 일지를 제공한 조건이다. 국내 데이터로 B를 평가할 수 있는 수면의 질 정답은 없다. MeLiDos 최종 평가는 이 실행에서 모델·설정 선정에 사용하지 않은 기존 37명 분할에 해당한다.

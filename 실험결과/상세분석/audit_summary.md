# 모델 검증 및 오차 분석 결과

## 검증 범위
원본 선택 레코드부터 전처리 데이터를 다시 구성하고, 저장된 예측값에서 평가 지표와 참가자 단위 10,000회 재표집 구간을 재계산했다. 저장된 MLP의 결측 대체·표준화와 가중치를 불러와 NumPy로 순전파를 독립 구현했다. 학습이나 Test를 이용한 설정 변경은 하지 않았다.

검증 체크 15개 통과. 저장 모델 비교 8개 통과. 최대 예측 차이 0.00001526.
완전한 원본 팀 ZIP 재구성 검증: 통과.
원래 PyTorch verify.py 및 모델 학습 전체는 이 환경에서 실행하지 않았다. NumPy 검증과 원래 학습 재현을 구분해야 한다.

## 데이터 관찰 기간
                   records            span_days           
                       min median max       min median max
split                                                     
test_korea              28   28.0  28        28   28.0  28
train_melidos            2    7.0   9         4    7.0  11
validation_melidos       6    7.0   9         6    7.0   9

## 주요 최종 성능
task         dataset     role    n  people  macro_mae      rmse  bias_pred_minus_actual  within_tolerance
   A       korea_all baseline 1372      49  21.205904 39.011270               -3.416545          0.564140
   A       korea_all selected 1372      49  21.308364 37.062104               -4.403738          0.400146
   A     korea_last7 baseline  343      49  20.058309 36.174343               -2.279883          0.606414
   A     korea_last7 selected  343      49  20.248309 34.093192               -3.587508          0.405248
   A melidos_holdout baseline  259      37  40.910730 79.358206               -6.866795          0.335907
   A melidos_holdout selected  259      37  38.135294 78.511662              -19.541970          0.308880
   B melidos_holdout baseline  259      37   0.750262  0.893573                0.135540          0.749035
   B melidos_holdout selected  259      37   0.699844  0.849404                0.093801          0.760618

## 실제 입면 지연 구간별 분석
task         dataset     role latency_band   n  people  macro_mae        mae       rmse  bias_pred_minus_actual  within_tolerance  actual_mean  predicted_mean
   A       korea_all baseline         0-10 468      44  23.901361  11.742521  22.145799               10.650641          0.694444     5.352564       16.003205
   A       korea_all baseline        10-30 541      47  17.148249  12.983364  25.273257                3.935305          0.683919    24.107209       28.042514
   A       korea_all baseline        30-60 201      40  28.565526  25.880597  34.900855               -7.800995          0.323383    51.194030       43.393035
   A       korea_all baseline          60+ 162      31  84.175713  70.203704  88.474904              -63.166667          0.086420   126.919753       63.753086
   A       korea_all selected         0-10 468      44  22.781034  16.620154  19.811764               16.620154          0.213675     5.352564       21.972718
   A       korea_all selected        10-30 541      47  10.383769   8.277741  13.380810                3.661328          0.744917    24.107209       27.768536
   A       korea_all selected        30-60 201      40  21.926472  20.298382  23.198805              -14.486353          0.208955    51.194030       36.707677
   A       korea_all selected          60+ 162      31  85.776911  79.621050  96.091739              -79.562855          0.024691   126.919753       47.356898
   A     korea_last7 baseline         0-10 115      34  15.230742   9.434783  16.570010                7.713043          0.773913     5.878261       13.591304
   A     korea_last7 baseline        10-30 134      41  20.172532  14.044776  28.212414                6.179104          0.701493    24.156716       30.335821
   A     korea_last7 baseline        30-60  56      29  29.379310  22.696429  31.496882               -7.446429          0.392857    52.392857       44.946429
   A     korea_last7 baseline          60+  38      16  80.592708  69.526316  81.927118              -54.736842          0.078947   124.394737       69.657895
   A     korea_last7 selected         0-10 115      34  17.672279  15.090641  17.239961               15.090641          0.208696     5.878261       20.968902
   A     korea_last7 selected        10-30 134      41  12.059726   9.249400  15.606482                5.073873          0.738806    24.156716       29.230589
   A     korea_last7 selected        30-60  56      29  22.738066  19.594279  22.676287              -13.834913          0.250000    52.392857       38.557944
   A     korea_last7 selected          60+  38      16  83.011235  75.606507  89.306007              -75.554810          0.052632   124.394737       48.839927
   A melidos_holdout baseline         0-10  52      20  27.480000  27.451923  50.626413               26.759615          0.461538     6.557692       33.317308
   A melidos_holdout baseline        10-30  84      32  37.012760  24.857143  61.718350               19.404762          0.488095    22.797619       42.202381
   A melidos_holdout baseline        30-60  59      27  27.880776  24.194915  39.157602                3.805085          0.322034    44.559322       48.364407
   A melidos_holdout baseline          60+  64      23  84.149793  90.226562 130.347987              -78.507812          0.046875   150.640625       72.132812
   A melidos_holdout selected         0-10  52      20  23.745751  23.178740  30.183186               23.178740          0.096154     6.557692       29.736433
   A melidos_holdout selected        10-30  84      32  18.346868  13.398179  24.500109               11.068463          0.595238    22.797619       33.866082
   A melidos_holdout selected        30-60  59      27  13.766689  13.920629  17.845353               -8.027303          0.406780    44.559322       36.532019
   A melidos_holdout selected          60+  64      23  94.379422 107.054484 152.064609             -105.043824          0.015625   150.640625       45.596801

구간은 오른쪽 끝을 포함한다: [0,10], (10,30], (30,60], (60,무한대). 그룹 그래프는 기록별 MAE이며 주 성능표는 참가자 평균 MAE다.

## 실제 수면의 질별 분석
task         dataset     role  actual   n  people  macro_mae      mae     rmse  bias_pred_minus_actual  within_tolerance  actual_mean  predicted_mean
   B melidos_holdout baseline     1.0   2       2   2.583417 2.583417 2.583417                2.583417          0.000000          1.0        3.583417
   B melidos_holdout baseline     2.0  38      22   1.583417 1.583417 1.583417                1.583417          0.000000          2.0        3.583417
   B melidos_holdout baseline     3.0  86      31   0.583417 0.583417 0.583417                0.583417          1.000000          3.0        3.583417
   B melidos_holdout baseline     4.0 108      33   0.416583 0.416583 0.416583               -0.416583          1.000000          4.0        3.583417
   B melidos_holdout baseline     5.0  25      15   1.416583 1.416583 1.416583               -1.416583          0.000000          5.0        3.583417
   B melidos_holdout selected     1.0   2       2   1.482853 1.482853 1.521899                1.482853          0.000000          1.0        2.482853
   B melidos_holdout selected     2.0  38      22   1.457206 1.431055 1.479530                1.431055          0.157895          2.0        3.431055
   B melidos_holdout selected     3.0  86      31   0.551350 0.552812 0.597702                0.515899          1.000000          3.0        3.515899
   B melidos_holdout selected     4.0 108      33   0.390426 0.417681 0.488403               -0.417681          0.972222          4.0        3.582319
   B melidos_holdout selected     5.0  25      15   1.309976 1.292368 1.304837               -1.292368          0.000000          5.0        3.707632

## 해석에 필요한 제한
- 37명 자료를 팀이 과거 모델 선택에 사용했는지는 코드만으로 확인할 수 없다. 확인 전에는 “별도 참가자 평가”로 표기한다.
- 국내 마지막 7일은 이전까지의 실제 수면일지를 매일 제공받는 조건이다. 앞 21일이 학습용 Train이라는 뜻은 아니다.
- A 국내 MAE와 ±10분 비율에서는 개인 과거 중앙값 기준선이 우세하다. RMSE만 낮아진 것으로 전체 개선을 주장하지 않는다.
- B 최종 입력 제외 비교는 알고리즘도 다르다. 수면 구간 길이 변수만의 비교는 b_same_mlp_ablation_cv.csv를 사용한다.
- 지역 및 작은 점수별 그룹은 소규모 탐색 분석이다. 지역 원인, 인과관계, 추천 치료 효과를 확정하지 않는다.
- 기존 계획의 early stopping은 실제 mlp.py에 구현되어 있지 않다. 실제로는 60/160 epoch를 교차검증으로 선택했다.
- 계획한 A 수면 길이 이력 제외 실험은 동봉 CV 결과에 없다. 완료한 실험으로 표시하지 않는다.
- 관찰된 수면 구간으로 B 점수를 예측한 결과는 후보 추천 효과나 개인 최적 수면 시간을 입증하지 않는다.

## 파일
- verified_metrics.csv: 저장된 모든 주 평가와 사후 트리 비교 지표 재검산
- verified_paired_intervals.csv: MLP 대비 기준선 및 입력 제외 모델의 참가자 단위 불확실성
- latency_bands.csv / quality_bands.csv / history_bands.csv / site_errors.csv: 하위집단별 오차
- audit_report.json: 검증 결과, 버전, 원본 SHA-256
- ../figures: 완전한 평가 행을 사용한 PNG 그래프 9개
- ../scripts/analyze_results.py: 본 분석의 재실행 코드

모든 분석의 자료 출처: model_package/data, model_package/results, model_package/tree_comparison, model_package/source_data.
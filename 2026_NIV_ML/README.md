# NIV 발생 여부 분류 ML 모형 (Threshold 0 ~ 0.20)

## 개요
- 데이터: `data/NIV_toxin_data.xlsx` (n = 80)
- 반응변수: `NIV (Y) > t` 이면 1(발생), 아니면 0 — t = 0.00, 0.02, …, 0.20 (11개)
- 설명변수(7): `FHB_incidence_predicted`, `temp/rhum/prcp_before_hd`, `temp/rhum/prcp_after_flower`
- 모형(8): Logistic(Elastic-net), SVM-RBF, KNN, MLP, RandomForest, ExtraTrees, XGBoost, LightGBM

## 방법
- **Nested CV**: outer = 5-fold × 3 repeats (성능평가), inner = 5-fold (Optuna TPE 튜닝, 목적함수 AUC)
  - 튜닝 trial: 빠른 모형 60회/outer fold, RF·ET·MLP 24회/outer fold
  - 탐색공간: 정규화 강도, penalty 종류, 트리 깊이·leaf 크기·feature 비율, 학습률, subsample, class_weight, 강수량 log 변환 여부 등
- **평가지표**: AUC, Brier score, Brier Skill Score (BSS = 1 − Brier / Brier_ref, Brier_ref = 유병률 상수예측)
  - repeat별 pooled OOF 예측으로 계산 후 평균 ± SD
- **최종모형**: 전체 데이터로 재튜닝(150회, 느린 모형 60회) 후 `results/models/*.joblib` 저장
- 최적모형 선택: AUC 최고(0.005 이내 동률 시 Brier 낮은 모형)

## 결과 — threshold별 최적 모형 (nested CV)

| Threshold | 양성/음성 | 최적 모형 | AUC (±SD) | Brier | BSS |
|---|---|---|---|---|---|
| 0.00 | 45/35 | KNN | 0.849 ± 0.014 | 0.158 | 0.356 |
| 0.02 | 41/39 | KNN | 0.839 ± 0.012 | 0.165 | 0.338 |
| 0.04 | 31/49 | LightGBM | **0.875** ± 0.030 | 0.137 | **0.425** |
| 0.06 | 27/53 | XGBoost | 0.810 ± 0.030 | 0.161 | 0.279 |
| 0.08 | 21/59 | KNN | 0.831 ± 0.021 | 0.136 | 0.295 |
| 0.10 | 20/60 | XGBoost | 0.864 ± 0.006 | **0.115** | 0.388 |
| 0.12 | 20/60 | XGBoost | 0.864 ± 0.006 | **0.115** | 0.388 |
| 0.14 | 18/62 | RandomForest | 0.863 ± 0.010 | 0.142 | 0.188 |
| 0.16 | 17/63 | XGBoost | **0.880** ± 0.013 | 0.120 | 0.282 |
| 0.18 | 16/64 | XGBoost | 0.847 ± 0.041 | 0.130 | 0.186 |
| 0.20 | 14/66 | XGBoost | 0.838 ± 0.044 | 0.120 | 0.171 |

모형별 평균(11개 threshold): XGBoost AUC 0.841 / Brier 0.139 / BSS 0.29 (가장 안정적),
RandomForest 0.843 / 0.147, KNN 0.830 / 0.141 … Logistic 0.772, MLP 0.743 (최하위).

### 해석
- 대부분 threshold에서 **XGBoost**가 AUC·Brier 모두 상위권 → 단일 모형을 고른다면 XGBoost 권장.
- t = 0.10 과 0.12 는 0.10 < NIV ≤ 0.12 인 표본이 없어 라벨이 동일 → 결과도 동일.
- t ≥ 0.14 에서는 양성 사례가 14~18개로 적어 AUC SD가 커지고 BSS가 낮아짐(불확실성 증가).
- 분류 목적이면 t = 0.04 (BSS 0.43) 또는 t = 0.10 (Brier 0.115, AUC SD 최소)이 성능·안정성 측면에서 유리.
- n = 80 의 소표본이므로 외부(다른 연도/지역) 데이터 검증이 필요.

## 산출물 (`results/`)
- `nested_cv_results.csv` — 전체 threshold × 모형 성능
- `best_model_per_threshold.csv` / `best_brier_model_per_threshold.csv` — 최적 모형 (AUC 기준 / Brier 기준)
- `final_hyperparameters.json` — 전체 데이터 재튜닝 하이퍼파라미터
- `performance_by_threshold.png`, `best_model_roc.png`, `best_model_calibration.png`
- `models/thr_<t>_<model>.joblib` — 최종 학습모형 (88개), `oof/` — out-of-fold 예측확률

## 실행
```bash
pip install -r requirements.txt
python niv_threshold_ml.py                    # 전체 (4코어 약 65분)
python niv_threshold_ml.py --objective brier  # Brier 기준 튜닝
python niv_threshold_ml.py --quick            # 빠른 테스트
```
모형 사용 예:
```python
import joblib, pandas as pd, sys; sys.path.insert(0, "2026_NIV_ML")
m = joblib.load("2026_NIV_ML/results/models/thr_0.10_XGBoost.joblib")
p = m.predict_proba(new_df[FEATURES].to_numpy(float))[:, 1]   # NIV > 0.10 확률
```

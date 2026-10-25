"""고추탄저병 설문 ML 분석 공통 함수.

- 데이터 로드 / 설명변수 세트 정의
- 학습 fold 안에서 변수 중요도 순위 계산 (XGB·RF SHAP + 정규화 선형모형 계수의 평균 순위)
- top-k 변수로 모델 학습·예측
"""
import warnings

import numpy as np
import pandas as pd
import shap
from scipy.special import logit, expit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

Y1, Y2 = "Y1 발생라벨", "Y2 피해도_전체"
STAGE2 = ["초발_DOY", "정식_초발_경과일", "이환과_즉시제거"]
WEATHER = ["6월_평균기온", "6월_평균상대습도", "6월_누적강수량", "6월_강우일수",
           "7~8월_평균기온", "7~8월_평균상대습도", "7~8월_누적강수량", "7~8월_강우일수",
           "6월_IR면적", "7~8월_IR면적"]
# 노출값 구간: 분포가 0 근처와 1 근처로 양극화되어 있어 4단계 순서형으로 변환
EXPO_BINS = [-np.inf, 0.05, 0.5, 0.95, np.inf]
SEV_BINS = [0, 0.02, 0.2, 1.01]           # 피해도 3단계: <2%, 2~20%, >=20%
SEV_LABELS = ["경(<2%)", "중(2~20%)", "심(≥20%)"]


def load(path):
    d = pd.read_excel(path, "모델링_데이터셋")
    d["노출값_등급"] = pd.cut(d["노출값"], EXPO_BINS, labels=False).astype(int)
    return d


def feature_sets(d, stage2):
    """A: v7(기상·IR 제외)+노출값, A_bin: 노출값을 4단계 등급으로, B: A+기상8+IR2."""
    allx = list(d.loc[:, "품종저항성_ord":"노출값"].columns)
    base = [c for c in allx if c not in WEATHER and (stage2 or c not in STAGE2)]
    return {
        "A": base,
        "A_bin": [c if c != "노출값" else "노출값_등급" for c in base],
        "B": base + WEATHER,
    }


# ---------------- 모델 ----------------
def make_model(name, task, seed=0):
    imp = SimpleImputer(strategy="median")
    if task == "cls":
        if name == "LR":
            return make_pipeline(imp, StandardScaler(), LogisticRegressionCV(
                Cs=12, penalty="elasticnet", solver="saga", l1_ratios=[0.2, 0.5, 0.9],
                cv=5, scoring="roc_auc", max_iter=5000, random_state=seed))
        if name == "RF":
            return make_pipeline(imp, RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                random_state=seed, n_jobs=1))
        if name == "XGB":
            return make_pipeline(imp, XGBClassifier(
                n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=2, reg_lambda=1.0,
                random_state=seed, n_jobs=1, verbosity=0))
    else:  # reg: 변환된 피해도(logit)를 예측
        if name == "EN":
            return make_pipeline(imp, StandardScaler(), ElasticNetCV(
                l1_ratio=[0.2, 0.5, 0.9], cv=5, max_iter=20000, random_state=seed))
        if name == "RF":
            return make_pipeline(imp, RandomForestRegressor(
                n_estimators=400, min_samples_leaf=3, max_features=0.5,
                random_state=seed, n_jobs=1))
        if name == "XGB":
            return make_pipeline(imp, XGBRegressor(
                n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=2, reg_lambda=1.0,
                random_state=seed, n_jobs=1, verbosity=0))
    raise ValueError(name)


CLS_MODELS = ["LR", "RF", "XGB"]
REG_MODELS = ["EN", "RF", "XGB"]


def fit_predict(name, task, Xtr, ytr, Xte, seed=0):
    m = make_model(name, task, seed).fit(Xtr, ytr)
    if task == "cls":
        return m, m.predict_proba(Xte)[:, 1]
    return m, m.predict(Xte)


# ---------------- 변수 중요도 ----------------
def shap_abs(model, X, task):
    """파이프라인 마지막 단계(트리모형)의 mean |SHAP|."""
    est = model[-1]
    Xi = model[:-1].transform(X)
    sv = shap.TreeExplainer(est).shap_values(Xi)
    sv = np.asarray(sv)
    if sv.ndim == 3:            # RF 분류: (n, p, class)
        sv = sv[..., 1]
    return np.abs(sv).mean(0)


def linear_abs(model):
    coef = np.ravel(model[-1].coef_)
    return np.abs(coef)


def rank_features(X, y, task, seed=0):
    """세 모델 중요도의 평균 순위(작을수록 중요). 학습 fold 안에서만 호출."""
    names = REG_MODELS if task == "reg" else CLS_MODELS
    ranks = []
    for n in names:
        m = make_model(n, task, seed).fit(X, y)
        imp = linear_abs(m) if n in ("LR", "EN") else shap_abs(m, X, task)
        ranks.append(pd.Series(imp, index=X.columns).rank(ascending=False))
    return pd.concat(ranks, axis=1).mean(1).sort_values()


# ---------------- 피해도 변환 ----------------
def sev_to_logit(y):
    """Smithson-Verkuilen 압축 후 logit. y=0/1 도 유한값이 되도록."""
    n = 67
    return logit((np.asarray(y) * (n - 1) + 0.5) / n)


def logit_to_sev(z):
    n = 67
    return np.clip((expit(z) * n - 0.5) / (n - 1), 0, 1)

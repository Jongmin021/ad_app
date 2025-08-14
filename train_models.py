# train_models.py
import os, json, joblib
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import optuna
from variance_rule import VarianceRuleDetector

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ---------- 1) 데이터 로드 ----------
df_final_vif = pd.read_csv("/Users/t2023-m0056/Downloads/df_final_vif.csv")  # 분류
df_mean = pd.read_csv("/Users/t2023-m0056/Downloads/df_mean.csv")  # 변동성룰
df_thresh_z = pd.read_csv(
    "/Users/t2023-m0056/Downloads/df_thresh_z.csv"
)  # 변동성룰 임계

# ---------- 2) LightGBM: 전체피처로 1차 학습 → 상위15 선정 ----------
# y
y = df_final_vif["ClusterLabel"].astype(int)

# 날짜 후보 식별 및 파싱
date_like = [
    c
    for c in df_final_vif.columns
    if any(k in c.lower() for k in ["date", "time", "datetime", "timestamp"])
    or any(k in c for k in ["일자", "날짜", "시각"])
]
for c in date_like:
    df_final_vif[c] = pd.to_datetime(df_final_vif[c], errors="coerce")

# X 구성: 날짜 컬럼 제외(간단/안전)
X = df_final_vif.drop(columns=["ClusterLabel", *date_like], errors="ignore")

# 전 컬럼 수치화 → 전부 NaN 컬럼 제거 → 결측 대체
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median(numeric_only=True))


X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
sw = compute_sample_weight(class_weight="balanced", y=y_tr)

base = LGBMClassifier(
    objective="multiclass",
    num_class=len(sorted(y.unique())),
    metric="multi_logloss",
    random_state=42,
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=5.0,
)
base.fit(X_tr, y_tr, sample_weight=sw)

gain = base.booster_.feature_importance(importance_type="gain")
feat_names = base.booster_.feature_name()
fi = pd.DataFrame({"f": feat_names, "g": gain}).sort_values("g", ascending=False)
top15 = fi.head(15)["f"].tolist()

# ---------- 3) Optuna: 상위15 피처로 최적화 ----------
X_tr2 = X_tr[top15]
X_te2 = X_te[top15]
sw2 = sw


def objective(trial):
    params = dict(
        objective="multiclass",
        num_class=len(sorted(y.unique())),
        metric="multi_logloss",
        boosting_type="gbdt",
        random_state=42,
        verbosity=-1,
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        num_leaves=trial.suggest_int("num_leaves", 15, 63),
        subsample=trial.suggest_float("subsample", 0.5, 0.9),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 30),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1.0, 10.0),
    )
    m = LGBMClassifier(**params).fit(X_tr2, y_tr, sample_weight=sw2)
    pred = m.predict(X_te2)
    return f1_score(y_te, pred, average="weighted")


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
clf15 = LGBMClassifier(
    **best_params,
    objective="multiclass",
    num_class=len(sorted(y.unique())),
    metric="multi_logloss",
    boosting_type="gbdt",
    random_state=42
)
clf15.fit(X_tr2, y_tr, sample_weight=sw2)

# ---------- 4) (순서1) 모델 저장 ----------
joblib.dump(clf15, MODELS / "lgbm_top15.pkl")
(MODELS / "selected_features_top15.json").write_text(
    json.dumps(top15, ensure_ascii=False, indent=2)
)

# ---------- 5) 변동성룰 학습 ----------
vr = VarianceRuleDetector(k_min_hits=1).fit_from_tables(df_mean, df_thresh_z)

# ---------- 6) (순서2) 모델 저장 ----------
joblib.dump(vr, MODELS / "variance_rule.pkl")

print("✅ Saved:")
print(" - models/lgbm_top15.pkl")
print(" - models/selected_features_top15.json")
print(" - models/variance_rule.pkl")

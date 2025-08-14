import pandas as pd

# ===== 0) 데이터 로드 =====
df_sensor_ml_roc = pd.read_csv(
    "/Users/t2023-m0056/Desktop/파일/df_sensor_ml_roc (1).csv"
)
df_unique_vars = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_unique_vars.csv")

# 오버레이 저장 정책
SAVE_ONLY_DEFECTS = False  # True: 불량(PredCluster>0)만 저장 / False: 정상 포함 저장


# ===== 1) 전처리 =====
def preprocess_sensor_data(df_sensor):
    import numpy as np
    import pandas as pd

    # 증강 플래그 없으면 0으로 생성(안전)
    if "is_augmented" not in df_sensor.columns:
        df_sensor = df_sensor.copy()
        df_sensor["is_augmented"] = 0

    target_cols = ["pH", "Temp", "Current", "Voltage"]  # Power, Resistance는 계산

    df_sensor = df_sensor.copy()
    df_sensor["Power"] = df_sensor["Current"] * df_sensor["Voltage"]
    df_sensor["Resistance"] = df_sensor["Voltage"] / df_sensor["Current"].replace(
        0, np.nan
    )

    for col in target_cols + ["Power", "Resistance"]:
        df_sensor[f"{col}_ma6"] = df_sensor.groupby(["Date", "Lot"])[col].transform(
            lambda x: x.rolling(window=6, min_periods=6).mean()
        )
        df_sensor[f"{col}_std6"] = df_sensor.groupby(["Date", "Lot"])[col].transform(
            lambda x: x.rolling(window=6, min_periods=6).std()
        )
        df_sensor[f"{col}_diff"] = df_sensor.groupby(["Date", "Lot"])[col].transform(
            lambda x: x.diff()
        )
        df_sensor[f"{col}_lag1"] = df_sensor.groupby(["Date", "Lot"])[col].transform(
            lambda x: x.shift(1)
        )
        df_sensor[f"{col}_lag6"] = df_sensor.groupby(["Date", "Lot"])[col].transform(
            lambda x: x.shift(6)
        )

    df_sensor = df_sensor.dropna().reset_index(drop=True)

    exclude_cols = (
        ["Index", "Lot", "Date", "ClusterLabel", "is_augmented"]  # ← 메타 제외
        + [f"{col}_lag1" for col in target_cols]
        + [f"{col}_lag6" for col in target_cols]
    )
    num_cols = [
        c
        for c in df_sensor.select_dtypes(include="number").columns
        if c not in exclude_cols
    ]

    def min_max_range(x):
        return x.max() - x.min()

    df_agg = df_sensor.groupby(["Date", "Lot"])[num_cols].agg(
        ["mean", "std", "min", "max", min_max_range]
    )
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index()

    mean_cols = [col for col in df_agg.columns if col.endswith("_mean")]
    df_agg = df_agg.sort_values(by=["Date", "Lot"]).reset_index(drop=True)
    for col in mean_cols:
        df_agg[f"{col}_lag1"] = df_agg[col].shift(1)
        df_agg[f"{col}_diff1"] = df_agg[col] - df_agg[f"{col}_lag1"]

    df_target = (
        df_sensor.groupby(["Date", "Lot"])["ClusterLabel"]
        .agg(["max", "first"])
        .reset_index()
        .rename(columns={"max": "ClusterLabel", "first": "ExampleLabel"})
    )
    df_target["IsDefect"] = (df_target["ClusterLabel"] > 0).astype(int)

    # ★ 증강 플래그를 LOT 단위로 보존
    df_flag = df_sensor.groupby(["Date", "Lot"])["is_augmented"].max().reset_index()

    df_final = pd.merge(
        df_agg,
        df_target[["Date", "Lot", "ClusterLabel", "IsDefect"]],
        on=["Date", "Lot"],
    )
    df_final = pd.merge(df_final, df_flag, on=["Date", "Lot"], how="left")
    df_final = df_final.dropna().reset_index(drop=True)
    return df_final


# ===== 2) 증강 함수 =====
def sampling_with_unique_lots(data, num_augment=10, noise_level=0.02):
    import pandas as pd
    import numpy as np

    sampling_data = []
    base_lot = data["Lot"].max() + 1  # 증강 Lot 번호 시작점

    for i in range(num_augment):
        sampled = data.copy()
        sampled["is_augmented"] = 1  # ★ 증강 표시
        shift = np.random.randint(1, 6) * 5
        sampled["DateTime"] = sampled["DateTime"] + pd.to_timedelta(shift, unit="s")

        for col in ["pH", "Temp", "Current", "Voltage"]:
            sampled[col] += np.random.normal(0, noise_level, size=len(sampled))

        sampled["Lot"] = sampled["Lot"] + base_lot + i
        sampling_data.append(sampled)

    return pd.concat(sampling_data, ignore_index=True)


# ===== 3) 데이터 분리/전처리 =====
# 불량/정상 원본에 플래그 0 부여
defective_data = df_sensor_ml_roc[
    df_sensor_ml_roc["ClusterLabel"].isin([1, 2, 3])
].copy()
defective_data["DateTime"] = pd.to_datetime(defective_data["DateTime"], errors="coerce")
defective_data["is_augmented"] = 0  # ★ 원본 불량

normal_data = df_sensor_ml_roc[df_sensor_ml_roc["ClusterLabel"] == 0].copy()
normal_data["is_augmented"] = 0  # ★ 원본 정상

# 증강(불량만 증강)
augmented_defective_data = sampling_with_unique_lots(defective_data, num_augment=10)

# 전처리(집계)
defective_final_orig = preprocess_sensor_data(defective_data)  # ★ 원본 불량
normal_final = preprocess_sensor_data(normal_data)  # ★ 원본 정상
augmented_final = preprocess_sensor_data(augmented_defective_data)  # ★ 증강 불량

# 학습 테이블(원본 정상 + 원본 불량 + 증강 불량 모두 포함: 학습 안정성 목적)
df_final_augmented = pd.concat(
    [normal_final, defective_final_orig, augmented_final], ignore_index=True
)

# ===== 4) 유의 변수 선택 + 메타 유지 =====
selected_cols = df_unique_vars["유의한_변수"].unique().tolist()
meta_cols = [
    "Date",
    "Lot",
    "ClusterLabel",
    "IsDefect",
    "is_augmented",
]  # ★ is_augmented 유지

available_cols = [col for col in selected_cols if col in df_final_augmented.columns]
df_final_filtered = df_final_augmented[meta_cols + available_cols].copy()

# 모델 입력에서 메타 일부 제거(행 정합 보존 위해 index는 유지)
df_final_filtered = df_final_filtered.drop(columns=["Date", "Lot", "IsDefect"])

# ===== 5) VIF 기반 변수 축소 =====
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

VIF_THRESHOLD = 10
CORR_THRESHOLD = 0.9
exclude_cols = [
    "Date",
    "Lot",
    "IsDefect",
    "ClusterLabel",
    "is_augmented",
]  # ★ 메타 제외
df_iter = df_final_filtered.copy()
iteration = 1


def calculate_vif(df, features):
    X = df[features].copy()
    X_const = add_constant(X, has_constant="add")
    vif = pd.DataFrame()
    vif["변수"] = X_const.columns[1:]
    vif["VIF"] = [
        variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])
    ]
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


while True:
    numeric_cols = df_iter.select_dtypes(include="number").columns
    features = [col for col in numeric_cols if col not in exclude_cols]
    if len(features) <= 1:
        break

    vif_data = calculate_vif(df_iter, features)
    high_vif_vars = vif_data[vif_data["VIF"] >= VIF_THRESHOLD]["변수"].tolist()
    if not high_vif_vars:
        break

    removed_vars = set()
    corr_matrix = df_iter[features].corr()
    for vif_var in high_vif_vars:
        high_corr_vars = corr_matrix[vif_var][
            corr_matrix[vif_var].abs() >= CORR_THRESHOLD
        ]
        high_corr_vars = high_corr_vars.drop(vif_var, errors="ignore")

        if high_corr_vars.empty:
            df_iter.drop(columns=[vif_var], errors="ignore", inplace=True)
            removed_vars.add(vif_var)
            break
        else:
            candidates = list(high_corr_vars.index) + [vif_var]
            best_var, best_corr = None, -1
            for var in candidates:
                max_corr = 0
                for c in [1, 2, 3]:
                    subset = df_iter[df_iter["ClusterLabel"].isin([0, c])]
                    try:
                        corr_val = abs(subset[["ClusterLabel", var]].corr().iloc[0, 1])
                        max_corr = max(max_corr, corr_val)
                    except Exception:
                        continue
                if max_corr > best_corr:
                    best_corr, best_var = max_corr, var
            to_remove = [v for v in candidates if v != best_var]
            df_iter.drop(columns=to_remove, errors="ignore", inplace=True)
            removed_vars.update(to_remove)
            break

    if not removed_vars:
        break
    iteration += 1

final_selected_columns = sorted(df_iter.columns.tolist())
df_final_vif = df_final_filtered[final_selected_columns].copy()

# ===== 6) 모델 학습 =====
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import optuna

top_15_features = [
    "Resistance_min_max_range",
    "pH_ma6_mean",
    "Resistance_ma6_min",
    "Resistance_ma6_min_max_range",
    "Power_diff_mean",
    "pH_std6_mean",
    "Power_ma6_std",
    "Power_std6_max",
    "Power_ma6_mean",
    "Resistance_std6_mean_diff1",
    "Power_min_max_range",
    "Resistance_diff_mean",
    "pH_ma6_max",
    "pH_std6_min",
    "Current_min_max_range",
]

# y 그대로 두되 X에서 메타 제거
X = df_final_vif.drop(["ClusterLabel", "is_augmented"], axis=1)
y = df_final_vif["ClusterLabel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_top15 = X_train[top_15_features]
X_test_top15 = X_test[top_15_features]
sample_weights_top15 = compute_sample_weight(class_weight="balanced", y=y_train)

X_tr, X_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
    X_train_top15,
    y_train,
    sample_weights_top15,
    test_size=0.2,
    stratify=y_train,
    random_state=42,
)


def objective(trial):
    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
    }
    model = LGBMClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=sw_tr)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average="weighted")


study_top15 = optuna.create_study(direction="maximize")
study_top15.optimize(objective, n_trials=30, show_progress_bar=True)

best_model_top15 = LGBMClassifier(
    **study_top15.best_params,
    objective="multiclass",
    num_class=len(np.unique(y_train)),
    metric="multi_logloss",
    boosting_type="gbdt",
    random_state=42,
)
best_model_top15.fit(X_train_top15, y_train, sample_weight=sample_weights_top15)

# ===== 6-1) ★★★★★ 앱 평가용 정답 데이터 저장 ★★★★★ =====
# 모델 학습에 사용된 데이터에서 원본(is_augmented==0) 데이터의 정답 라벨을 추출하여 저장합니다.
# 이것이 Streamlit 앱에서 성능을 평가할 때 사용해야 할 "진짜 정답지"입니다.
ground_truth_df = df_final_augmented.query("is_augmented == 0")[
    ["Date", "Lot", "ClusterLabel"]
].copy()
ground_truth_df.to_csv("ground_truth_for_app.csv", index=False)
# =========================================================

# ===== 7) 예측 저장 — 원본만(+옵션: 불량만) =====
X_all_top15 = df_final_vif[top_15_features]
pred_all = best_model_top15.predict(X_all_top15)

# df_final_filtered와 행 정합 사용 → 원본/증강 플래그 접근
meta_for_pred = df_final_augmented.loc[
    df_final_filtered.index, ["Date", "Lot", "is_augmented"]
].reset_index(drop=True)
assert len(meta_for_pred) == len(pred_all), "행 정합 불일치: 메타/예측 길이 확인"

pred_lot = meta_for_pred.assign(PredCluster=pred_all)

# ★ 원본(비증강)만 남김
pred_lot = pred_lot.query("is_augmented == 0").drop(columns=["is_augmented"])

# ★ (옵션) 불량만 저장
if SAVE_ONLY_DEFECTS:
    pred_lot = pred_lot.query("PredCluster != 0")

pred_lot = pred_lot.drop_duplicates(subset=["Date", "Lot"], keep="last")
pred_lot.to_csv("pred_lot.csv", index=False)

# (선택) 성능 확인용
train_pred_top15 = best_model_top15.predict(X_train_top15)
test_pred_top15 = best_model_top15.predict(X_test_top15)
# print("train/test done")

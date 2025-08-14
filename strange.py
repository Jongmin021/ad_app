import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

df_mean = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_mean (1).csv")
df_thresh_z = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_thresh_z (1).csv")
# =========================================================
# 0) 입력 전제
# - df_mean: ['Date','Lot','ClusterLabel', 변수들...] (ClusterLabel: 정상=0, 불량=1/2/3)
# - df_thresh_z: ['ClusterLabel','변수','임계값'] (+ 선택: 'direction' [+1/-1])
# =========================================================

# ---------------------------------------------------------
# 1) 설정 (🚨 이 부분의 임계값만 수정하세요)
# ---------------------------------------------------------
RND_DECIMALS = 2
LOG_EPS = 1e-6
WINDOW_SIZE = 6

# SPC 룰에 부여할 가중치 설정 (학습 시 사용한 가중치를 그대로 적용)
RULE_WEIGHTS = {
    "rule1": 5,
    "rule2": 4,
    "rule3": 3,
    "rule4": 2,
    "rule5": 2,
    "rule6": 2,
    "rule7": 1,
    "rule8": 1,
}

# ⭐ 학습을 통해 얻은 최적 임계값을 여기에 직접 입력합니다 ⭐
OPTIMAL_THRESHOLD_1 = 2.0  # 변동성 룰 모델의 최적 임계값 (예시)
OPTIMAL_THRESHOLD_2 = 7.0  # SPC 룰 모델의 최적 임계값 (예시)


# ---------------------------------------------------------
# 2) 데이터 준비 및 전처리
# ---------------------------------------------------------
df_mean_sorted = df_mean.sort_values(["Date", "Lot"]).reset_index(drop=True)
df_thresh_z_binary = df_thresh_z[df_thresh_z["ClusterLabel"] > 0].copy()
if "direction" not in df_thresh_z_binary.columns:
    df_thresh_z_binary["direction"] = 1
df_thresh_z_binary["effective_threshold"] = (
    df_thresh_z_binary["임계값"] * df_thresh_z_binary["direction"]
)
df_thresh_binary_final = df_thresh_z_binary.loc[
    df_thresh_z_binary.groupby("변수")["effective_threshold"].idxmin()
].reset_index(drop=True)
df_thresh_binary_final["임계값"] = df_thresh_binary_final["effective_threshold"]

vars_with_thresholds = sorted(
    set(df_thresh_binary_final["변수"]).intersection(df_mean_sorted.columns)
)
if not vars_with_thresholds:
    raise ValueError(
        "변동성룰에 사용할 변수가 없습니다. 'df_thresh_z'와 'df_mean'의 변수명을 확인하세요."
    )

df_all = df_mean_sorted.copy()
y_true = (df_all["ClusterLabel"] > 0).astype(int)
grouped = df_all.groupby(["Date", "Lot"])

df_normal = df_all[df_all["ClusterLabel"] == 0]
normal_stats = df_normal[vars_with_thresholds].agg(["mean", "std"]).T
normal_stats.columns = ["mean", "std"]


# ---------------------------------------------------------
# 3) 모델 함수 정의
# ---------------------------------------------------------
def run_variance_rule_model(df, cluster_thr_df):
    scores = []
    for _, row in df.iterrows():
        cur_score = 0
        for _, t in cluster_thr_df.iterrows():
            var, thr, d = t["변수"], t["임계값_tr"], t["direction"]
            if var in row.index and not pd.isna(row[var]) and not pd.isna(thr):
                if d * (row[var] - thr) > 0:
                    cur_score += 1
        scores.append(cur_score)
    return pd.Series(scores, index=df.index)


def run_spc_rule_model(df, vars_list, rule_weights, normal_stats):
    df_model_2 = df.copy()
    spc_scores = pd.DataFrame(
        np.zeros((len(df_model_2), len(vars_list))),
        index=df_model_2.index,
        columns=vars_list,
    )
    grouped = df_model_2.groupby(["Date", "Lot"])

    for var in vars_list:
        if var not in normal_stats.index:
            continue

        z_score = pd.Series(0.0, index=df_model_2.index)
        for _, group_df in grouped:
            if len(group_df) >= 2:
                ma = group_df[var].rolling(window=WINDOW_SIZE, min_periods=1).mean()
                std = group_df[var].rolling(window=WINDOW_SIZE, min_periods=1).std()
                group_z_score = (group_df[var] - ma) / std.replace(0, np.nan)
            else:
                mean = normal_stats.loc[var, "mean"]
                std = normal_stats.loc[var, "std"]
                group_z_score = (
                    (group_df[var] - mean) / std
                    if not pd.isna(std) and std != 0
                    else pd.Series(0.0, index=group_df.index)
                )
            z_score.loc[group_df.index] = group_z_score.fillna(0)

        spc_scores[var] += (np.abs(z_score) > 3).astype(int) * rule_weights["rule1"]
        spc_scores[var] += (
            ((np.abs(z_score.shift(1)) > 2) & (np.abs(z_score) > 2))
            | ((np.abs(z_score.shift(2)) > 2) & (np.abs(z_score) > 2))
        ).astype(int) * rule_weights["rule2"]
        spc_scores[var] += (
            (
                (np.abs(z_score.shift(1)) > 1)
                & (np.abs(z_score) > 1)
                & (np.abs(z_score.shift(2)) > 1)
                & (np.abs(z_score.shift(3)) > 1)
            )
            | (
                (np.abs(z_score.shift(2)) > 1)
                & (np.abs(z_score.shift(3)) > 1)
                & (np.abs(z_score.shift(4)) > 1)
                & (np.abs(z_score) > 1)
            )
        ).astype(int) * rule_weights["rule3"]

        # 🚨 이 부분의 코드가 수정되었습니다.
        spc_scores[var] += (
            z_score.rolling(window=9, center=False)
            .apply(lambda x: 1 if (x > 0).all() or (x < 0).all() else 0, raw=True)
            .fillna(0)
            .astype(int)
            * rule_weights["rule4"]
        )
        spc_scores[var] += (
            z_score.rolling(window=6, center=False)
            .apply(
                lambda x: (
                    1
                    if (pd.Series(x).diff().dropna() > 0).all()
                    or (pd.Series(x).diff().dropna() < 0).all()
                    else 0
                ),
                raw=True,
            )
            .fillna(0)
            .astype(int)
            * rule_weights["rule5"]
        )
        spc_scores[var] += (
            z_score.rolling(window=14, center=False)
            .apply(
                lambda x: (
                    1
                    if (
                        np.sign(pd.Series(x).diff().dropna())[1:]
                        * np.sign(pd.Series(x).diff().dropna())[:-1]
                        < 0
                    ).all()
                    else 0
                ),
                raw=True,
            )
            .fillna(0)
            .astype(int)
            * rule_weights["rule6"]
        )
        spc_scores[var] += (
            z_score.rolling(window=15, center=False)
            .apply(lambda x: 1 if (pd.Series(x).abs() < 1).all() else 0, raw=True)
            .fillna(0)
            .astype(int)
            * rule_weights["rule7"]
        )
        spc_scores[var] += (
            z_score.rolling(window=8, center=False)
            .apply(lambda x: 1 if (pd.Series(x).abs() > 1).all() else 0, raw=True)
            .fillna(0)
            .astype(int)
            * rule_weights["rule8"]
        )

    return spc_scores.sum(axis=1)


# ---------------------------------------------------------
# 4) 각 모델 실행 및 예측 점수 생성
# ---------------------------------------------------------
# 모델 1: 변동성 룰
df_all_scaled = df_all.copy()
transformers = {
    var: StandardScaler().fit(df_normal[[var]]) for var in vars_with_thresholds
}
df_thresh_tr = df_thresh_binary_final.copy()
df_thresh_tr["임계값_tr"] = df_thresh_tr.apply(
    lambda r: transformers[r["변수"]].transform([[r["임계값"]]])[0][0], axis=1
)
cluster_thr_df = df_thresh_tr[["변수", "임계값_tr", "direction"]].dropna(
    subset=["임계값_tr"]
)

for var in vars_with_thresholds:
    df_all_scaled[var] = transformers[var].transform(df_all_scaled[[var]])

score_model_1 = run_variance_rule_model(df_all_scaled, cluster_thr_df)

# 모델 2: SPC 룰
score_model_2 = run_spc_rule_model(
    df_all, vars_with_thresholds, RULE_WEIGHTS, normal_stats
)

# print(f"✔️ 변동성 룰 모델 최적 임계값: {OPTIMAL_THRESHOLD_1}")
# print(f"✔️ SPC 룰 모델 최적 임계값: {OPTIMAL_THRESHOLD_2}")

# ---------------------------------------------------------
# 5) 앙상블(투표) 및 최종 판정
# ---------------------------------------------------------
predictions_df = pd.DataFrame(index=df_all.index)
# 사전에 설정한 최적 임계값을 사용하여 예측
predictions_df["pred_model_1"] = (score_model_1 >= OPTIMAL_THRESHOLD_1).astype(int)
predictions_df["pred_model_2"] = (score_model_2 >= OPTIMAL_THRESHOLD_2).astype(int)
predictions_df["vote_score"] = predictions_df[["pred_model_1", "pred_model_2"]].sum(
    axis=1
)

df_all["Final_Flag"] = "정상"
df_all.loc[predictions_df["vote_score"] == 1, "Final_Flag"] = "경고"
df_all.loc[predictions_df["vote_score"] >= 2, "Final_Flag"] = "조치 필요"

# print("\n=== 앙상블 최종 판정 결과(샘플) ===")
# print(df_all[['Date','Lot','ClusterLabel','Final_Flag']].head(10))

# ---------------------------------------------------------
# 6) 최종 성능 평가
# ---------------------------------------------------------
y_pred_final = (predictions_df["vote_score"] >= 1).astype(int)
cm = confusion_matrix(y_true, y_pred_final, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred_final, average="binary", zero_division=0
)

# print("\n=== 앙상블 최종 성능 평가 ===")
# print("\n=== Confusion Matrix (0:정상, 1:불량) ===")
# print(pd.DataFrame(cm, index=['True_0','True_1'], columns=['Pred_0','Pred_1']))
# print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
# print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

# ===== LOT 단위 이진 판정 저장 =====
# row 단위 → LOT 단위로 집계 (한 번이라도 1이면 LOT=1)
row_results = df_all[["Date", "Lot"]].copy()
row_results["PredFlag"] = y_pred_final.values  # 0/1 (vote_score>=1)

# LOT별 이진 플래그
lot_flag = row_results.groupby(["Date", "Lot"], as_index=False).agg(
    PredFlag=("PredFlag", "max")
)

# (선택) LOT별 최대 점수도 같이 저장하고 싶다면:
lot_scores = pd.DataFrame(
    {
        "Date": df_all["Date"],
        "Lot": df_all["Lot"],
        "score_variance": score_model_1.values,
        "score_spc": score_model_2.values,
        "vote_score": predictions_df["vote_score"].values,
    }
)
lot_scores = lot_scores.groupby(["Date", "Lot"], as_index=False).agg(
    score_variance_max=("score_variance", "max"),
    score_spc_max=("score_spc", "max"),
    vote_score_max=("vote_score", "max"),
)

# 병합(원하면 빼도 됨)
lot_out = lot_flag.merge(lot_scores, on=["Date", "Lot"], how="left")

# 문자열로 저장(앱 호환)
lot_out["Date"] = pd.to_datetime(lot_out["Date"]).dt.date.astype(str)
lot_out["Lot"] = lot_out["Lot"].astype(str)

# 저장
out_path = "pred_binary_lot.csv"
lot_out.to_csv(out_path, index=False)
print(f"✔ saved: {out_path} ({lot_out.shape[0]} lots)")

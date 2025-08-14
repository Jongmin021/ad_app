import pandas as pd

df_mean = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_mean (1).csv")
df_thresh_z = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_thresh_z (1).csv")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

# =========================================================
# 0) 입력 전제
# - df_mean: ['Date','Lot','ClusterLabel', 변수들...] (ClusterLabel: 정상=0, 불량=1/2/3)
# - df_thresh_z: ['ClusterLabel','변수','임계값'] (+ 선택: 'direction' [+1/-1])
# =========================================================

# ---------------------------------------------------------
# 1) 스케일 기법 선택 모드 설정
#    'global' | 'per_var_map' | 'auto'
# ---------------------------------------------------------
scale_selection_mode = "auto"  # 필요 시 'global' 또는 'per_var_map'
global_method = "standard"  # 'global' 모드일 때 사용: 'standard'|'robust'|'log'
default_method = "standard"  # 'per_var_map' 모드의 기본값

# 변수별 수동 매핑(선택)
method_map = {
    # 예시) '전압': 'robust', '저항': 'standard', '온도': 'log'
}

# k-out-of-n 규칙 설정: 초과 변수 개수 >= k면 Flag=1
k_min_hits = 1

# ---------------------------------------------------------
# 2) 임계값 대상 변수 목록
# ---------------------------------------------------------
vars_with_thresholds = sorted(set(df_thresh_z["변수"]).intersection(df_mean.columns))


# ---------------------------------------------------------
# 3) 자동 선택 함수(정상군 기반)
#    - 강한 양의 왜도이며 비음수: log
#    - 이상치/강왜도: robust
#    - 그 외: standard
#    - 음수/0 존재 시 log 대신 robust 권장
# ---------------------------------------------------------
def choose_method_by_profile(s_normal, skew_th=1.0, outlier_q=0.99):
    s = s_normal.dropna()
    if len(s) < 50:
        return "robust"
    skew = s.skew()
    q01, q99 = s.quantile([0.01, outlier_q])
    q25, q75 = s.quantile([0.25, 0.75])
    iqr = q75 - q25
    outlier_like = (iqr == 0) or (q99 - q01 > 6 * (iqr if iqr > 0 else 1))
    has_nonpositive = (s <= 0).any()
    if abs(skew) > skew_th and not has_nonpositive:
        return "log"
    if outlier_like or abs(skew) > 2 * skew_th:
        return "robust"
    return "standard"


# ---------------------------------------------------------
# 4) 스케일러 학습(정상군 기준). 로그의 경우 음수/0 대응으로 log_shift 지원.
#    transformers[var] = ('kind', object_or_params_dict)
#    kind: 'standard' | 'robust' | 'log' | 'log_shift'
# ---------------------------------------------------------
def fit_transformers(
    df,
    variables,
    selection_mode="auto",
    global_method="standard",
    method_map=None,
    default_method="standard",
    log_eps=1e-6,
):
    transformers = {}
    df_normal = df[df["ClusterLabel"] == 0].copy()
    for var in variables:
        if var not in df.columns:
            continue
        x = df_normal[[var]].dropna().values
        # 선택 모드별 기법 결정
        if selection_mode == "global":
            method = global_method
        elif selection_mode == "per_var_map":
            method = (method_map or {}).get(var, default_method)
        else:  # 'auto'
            method = choose_method_by_profile(df_normal[var])

        # 학습
        if method == "standard":
            scaler = StandardScaler()
            if len(x) > 0:
                scaler.fit(x)
            else:
                scaler.mean_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
            transformers[var] = ("standard", scaler)

        elif method == "robust":
            scaler = RobustScaler()
            if len(x) > 0:
                scaler.fit(x)
            else:
                scaler.center_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
            transformers[var] = ("robust", scaler)

        elif method == "log":
            s = df_normal[var].dropna()
            has_nonpositive = (s <= 0).any()
            if has_nonpositive:
                # 음수/0 존재 → shift+log
                shift = float(-(s.min()) + log_eps) if len(s) > 0 else 0.0
                transformers[var] = ("log_shift", {"eps": log_eps, "shift": shift})
            else:
                transformers[var] = ("log", {"eps": log_eps})

        else:
            # 방어적: 미지정 → standard
            scaler = StandardScaler()
            if len(x) > 0:
                scaler.fit(x)
            else:
                scaler.mean_ = np.array([0.0])
                scaler.scale_ = np.array([1.0])
            transformers[var] = ("standard", scaler)
    return transformers


# ---------------------------------------------------------
# 5) 변환 함수(시리즈/스칼라)
# ---------------------------------------------------------
def transform_series(s, transformer):
    kind, obj = transformer
    if kind == "standard":
        return pd.Series(obj.transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
    if kind == "robust":
        return pd.Series(obj.transform(s.values.reshape(-1, 1)).ravel(), index=s.index)
    if kind == "log":
        return np.log1p(s + obj["eps"])
    if kind == "log_shift":
        return np.log1p(s + obj["shift"] + obj["eps"])
    return s


def transform_scalar(v, transformer):
    if pd.isna(v):
        return np.nan
    kind, obj = transformer
    if kind == "standard":
        mean = float(obj.mean_[0])
        scale = float(obj.scale_[0] or 1.0)
        return (v - mean) / scale
    if kind == "robust":
        center = float(getattr(obj, "center_", [0.0])[0])
        scale = float(getattr(obj, "scale_", [1.0])[0] or 1.0)
        return (v - center) / scale
    if kind == "log":
        return np.log1p(v + obj["eps"])
    if kind == "log_shift":
        return np.log1p(v + obj["shift"] + obj["eps"])
    return v


# ---------------------------------------------------------
# 6) 스케일러 학습 및 스케일 변환
# ---------------------------------------------------------
transformers = fit_transformers(
    df_mean,
    vars_with_thresholds,
    selection_mode=scale_selection_mode,
    global_method=global_method,
    method_map=method_map,
    default_method=default_method,
    log_eps=1e-6,
)

# 선택된 기법 기록(재현성)
method_records = {var: transformers[var][0] for var in transformers}
method_df = pd.DataFrame.from_dict(
    method_records, orient="index", columns=["scale_method"]
)
# method_df.to_csv('scale_method_map.csv', encoding='utf-8-sig')  # 필요 시 저장

# 데이터 변환
df_mean_tr = df_mean.copy()
excluded_vars = []  # 결측·비정상 등으로 제외된 변수 기록
for var in vars_with_thresholds:
    try:
        df_mean_tr[var] = transform_series(
            df_mean_tr[var].astype(float), transformers[var]
        )
    except Exception as e:
        excluded_vars.append((var, str(e)))

# 임계값 변환(+ 방향성 보정)
df_thresh_tr = df_thresh_z.copy()
if "direction" not in df_thresh_tr.columns:
    df_thresh_tr["direction"] = 1  # 기본: 값이 클수록 불량

df_thresh_tr["임계값_tr"] = df_thresh_tr.apply(
    lambda r: transform_scalar(r["임계값"], transformers.get(r["변수"])), axis=1
)

# 동일 변수에서 변환 실패 등 결측 발생 시 제외 표시
bad_thr_rows = df_thresh_tr["임계값_tr"].isna()
if bad_thr_rows.any():
    for _, rr in df_thresh_tr[bad_thr_rows].iterrows():
        excluded_vars.append((rr["변수"], "threshold_transform_nan"))

# ---------------------------------------------------------
# 7) 클러스터별 임계 맵 구성
#    {cluster: DataFrame[변수, 임계값_tr, direction]}
# ---------------------------------------------------------
cluster_thr_df_map = {
    c: g[["변수", "임계값_tr", "direction"]].dropna(subset=["임계값_tr"])
    for c, g in df_thresh_tr.groupby("ClusterLabel")
}

# ---------------------------------------------------------
# 8) 변동성룰 적용(k-out-of-n, 방향성 반영)
#    - 정상(0)은 규칙 미적용
#    - margin = direction * (value - threshold_tr)
#    - hits: margin > 0인 변수 개수
# ---------------------------------------------------------
flags, hits, max_margins, trigger_lists = [], [], [], []

for idx, row in df_mean_tr.iterrows():
    c = row["ClusterLabel"]
    if c == 0 or c not in cluster_thr_df_map:
        flags.append(0)
        hits.append(0)
        max_margins.append(np.nan)
        trigger_lists.append("")
        continue

    thr_df = cluster_thr_df_map[c]
    cur_hits = 0
    cur_triggers = []
    max_margin = -np.inf

    for _, t in thr_df.iterrows():
        var = t["변수"]
        thr = t["임계값_tr"]
        d = t["direction"]
        if var not in row.index:
            continue
        v = row[var]
        if pd.isna(v) or pd.isna(thr):
            continue
        margin = d * (v - thr)
        if margin > 0:
            cur_hits += 1
            cur_triggers.append(var)
        if pd.notna(margin):
            max_margin = max(max_margin, margin)

    flag = 1 if cur_hits >= k_min_hits else 0
    flags.append(flag)
    hits.append(cur_hits)
    max_margins.append(max_margin if cur_hits > 0 else np.nan)
    trigger_lists.append(",".join(cur_triggers))

df_mean_tr["VarianceRuleFlag"] = flags
df_mean_tr["VarianceRuleHits"] = hits
df_mean_tr["VarianceRuleMaxMargin"] = max_margins
df_mean_tr["VarianceRuleTriggers"] = trigger_lists

# ---------------------------------------------------------
# 9) 결과 및 로그 확인
# ---------------------------------------------------------
# print("=== 변동성룰 결과 (샘플) ===")
# print(df_mean_tr[['Date','Lot','ClusterLabel','VarianceRuleFlag','VarianceRuleHits','VarianceRuleMaxMargin','VarianceRuleTriggers']].head())

# if excluded_vars:
#    excl_df = pd.DataFrame(excluded_vars, columns=['variable','reason']).drop_duplicates()
#    print("\n[경고] 제외/오류 변수 목록:")
#    print(excl_df.head(20))

import numpy as np
import pandas as pd


# 1) 정상군 분포 프로파일링
def profile_normal_distribution(df_mean, variables, normal_label=0):
    df_norm = df_mean[df_mean["ClusterLabel"] == normal_label]
    rows = []
    for var in variables:
        if var not in df_norm.columns:
            continue
        s = pd.to_numeric(df_norm[var], errors="coerce").dropna()
        if len(s) == 0:
            rows.append(
                {
                    "변수": var,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "skew": np.nan,
                    "kurt": np.nan,
                    "min": np.nan,
                    "q01": np.nan,
                    "q25": np.nan,
                    "q50": np.nan,
                    "q75": np.nan,
                    "q99": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "nonpositive_exists": np.nan,
                    "zero_ratio": np.nan,
                    "outlier_span_vs_iqr": np.nan,
                }
            )
            continue

        q01, q25, q50, q75, q99 = s.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        iqr = q75 - q25
        outlier_span = q99 - q01
        rows.append(
            {
                "변수": var,
                "n": int(s.size),
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "skew": s.skew(),
                "kurt": s.kurt(),
                "min": s.min(),
                "q01": q01,
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "q99": q99,
                "max": s.max(),
                "iqr": iqr,
                "nonpositive_exists": bool((s <= 0).any()),
                "zero_ratio": float((s == 0).mean()),
                "outlier_span_vs_iqr": float(outlier_span / (iqr if iqr > 0 else 1.0)),
            }
        )
    prof = pd.DataFrame(rows)
    return prof


# 2) 프로파일 기반 스케일 기법 추천
def recommend_scale_method(
    profile_df, skew_th=1.0, outlier_span_mult=6.0, min_n_for_decision=50
):
    methods, reasons = [], []
    for _, r in profile_df.iterrows():
        n = r["n"]
        skew = r["skew"]
        iqr = r["iqr"]
        span_ratio = r["outlier_span_vs_iqr"]
        has_nonpos = bool(r["nonpositive_exists"])

        # 표본이 작으면 보수적으로 robust
        if (pd.isna(n)) or (n < min_n_for_decision):
            methods.append("robust")
            reasons.append("n<min → robust")
            continue

        # 왜도와 이상치 정도 판단
        if pd.notna(skew) and abs(skew) > 2 * skew_th:
            # 매우 강한 왜도
            if not has_nonpos:
                methods.append("log")
                reasons.append("|skew| 매우 큼 & 양수 → log")
            else:
                methods.append("robust")
                reasons.append("|skew| 매우 큼 & 음/0 존재 → robust")
            continue

        # 일반적 왜도
        if pd.notna(skew) and abs(skew) > skew_th and not has_nonpos:
            methods.append("log")
            reasons.append("|skew| 큼 & 양수 → log")
            continue

        # 이상치가 두드러지면 robust
        if pd.notna(span_ratio) and (
            span_ratio > outlier_span_mult or (pd.notna(iqr) and iqr == 0)
        ):
            methods.append("robust")
            reasons.append("outlier span 큼 또는 IQR=0 → robust")
            continue

        # 기본값
        methods.append("standard")
        reasons.append("분포 양호 → standard")

    rec = profile_df.copy()
    rec["recommended_method"] = methods
    rec["reason"] = reasons

    # 음수/0 존재하는데 log가 추천된 경우 → log_shift로 교정
    rec.loc[
        (rec["recommended_method"] == "log") & (rec["nonpositive_exists"]),
        "recommended_method",
    ] = "log_shift"
    return rec


# 3) 파이프라인 연결
#    - vars_with_thresholds: df_thresh_z['변수'] ∩ df_mean.columns (이미 보유)
prof_df = profile_normal_distribution(df_mean, vars_with_thresholds, normal_label=0)
rec_df = recommend_scale_method(
    prof_df, skew_th=1.0, outlier_span_mult=6.0, min_n_for_decision=50
)

# 변수→기법 매핑 생성
auto_method_map = dict(zip(rec_df["변수"], rec_df["recommended_method"]))

# 확인용 표
# print("=== 정상군 분포 요약 + 추천 스케일 기법 ===")
# print(rec_df[['변수','n','skew','iqr','nonpositive_exists','outlier_span_vs_iqr','recommended_method','reason']].sort_values('변수').head(30))

# 기존 fit_transformers를 'per_var_map' 모드로 호출하여 추천 기법을 그대로 적용
transformers = fit_transformers(
    df_mean,
    vars_with_thresholds,
    selection_mode="per_var_map",
    global_method="standard",
    method_map=auto_method_map,  # 자동 추천 결과 주입
    default_method="standard",
    log_eps=1e-6,
)

# 선택된 기법 기록(재현성)
method_records = {var: transformers[var][0] for var in transformers}
method_df = pd.DataFrame.from_dict(
    method_records, orient="index", columns=["scale_method"]
)
# method_df.to_csv('scale_method_map.csv', encoding='utf-8-sig')
# ============================================
# A) 데이터/임계값 동일 스케일 변환
# ============================================
# 1) 적용 변수 집합 재확인
vars_with_thresholds = sorted(set(df_thresh_z["변수"]).intersection(df_mean.columns))

# 2) 데이터 변환
RND_DECIMALS = 2  # 소수 둘째 자리
df_mean_tr = df_mean.copy()
excluded_vars = []
for var in vars_with_thresholds:
    try:
        df_mean_tr[var] = transform_series(
            pd.to_numeric(df_mean_tr[var], errors="coerce"), transformers[var]
        )
    except Exception as e:
        excluded_vars.append((var, f"data_transform_error: {e}"))

df_mean_tr[vars_with_thresholds] = df_mean_tr[vars_with_thresholds].round(RND_DECIMALS)

# 3) 임계값 변환
df_thresh_tr = df_thresh_z.copy()
if "direction" not in df_thresh_tr.columns:
    df_thresh_tr["direction"] = 1  # 기본: 클수록 불량

df_thresh_tr["임계값_tr"] = df_thresh_tr.apply(
    lambda r: transform_scalar(r["임계값"], transformers.get(r["변수"])), axis=1
)

bad_thr_rows = df_thresh_tr["임계값_tr"].isna()
if bad_thr_rows.any():
    for _, rr in df_thresh_tr[bad_thr_rows].iterrows():
        excluded_vars.append((rr["변수"], "threshold_transform_nan"))

# ============================================
# B) 클러스터별 임계 테이블 구축
# ============================================
cluster_thr_df_map = {
    c: g[["변수", "임계값_tr", "direction"]].dropna(subset=["임계값_tr"])
    for c, g in df_thresh_tr.groupby("ClusterLabel")
}

# 확인용(선택): 추천 스케일 기법 요약 + 임계값 비교
compare_df = df_thresh_tr.merge(
    rec_df[["변수", "recommended_method"]], on="변수", how="left"
)[
    ["ClusterLabel", "변수", "recommended_method", "임계값", "임계값_tr", "direction"]
].sort_values(
    ["ClusterLabel", "변수"]
)
# print("=== 임계값(원본 vs 변환) 및 추천 스케일 기법 ===")
# print(compare_df.head(20))

# ============================================
# C) 변동성룰 적용 (k-out-of-n, 방향성 반영)
# ============================================
k_min_hits = 1  # 필요 시 조정

flags, hits, max_margins, trigger_lists = [], [], [], []
for idx, row in df_mean_tr.iterrows():
    c = row["ClusterLabel"]
    if c == 0 or c not in cluster_thr_df_map:
        flags.append(0)
        hits.append(0)
        max_margins.append(np.nan)
        trigger_lists.append("")
        continue

    thr_df = cluster_thr_df_map[c]
    cur_hits = 0
    cur_triggers = []
    max_margin = -np.inf

    for _, t in thr_df.iterrows():
        var = t["변수"]
        thr = t["임계값_tr"]
        d = t["direction"]
        if var not in row.index:
            continue
        v = row[var]
        if pd.isna(v) or pd.isna(thr):
            continue
        margin = d * (v - thr)
        if margin > 0:
            cur_hits += 1
            cur_triggers.append(var)
        if pd.notna(margin):
            max_margin = max(max_margin, margin)

    flag = 1 if cur_hits >= k_min_hits else 0
    flags.append(flag)
    hits.append(cur_hits)
    max_margins.append(max_margin if cur_hits > 0 else np.nan)
    trigger_lists.append(",".join(cur_triggers))

df_mean_tr["VarianceRuleFlag"] = flags
df_mean_tr["VarianceRuleHits"] = hits
df_mean_tr["VarianceRuleMaxMargin"] = max_margins
df_mean_tr["VarianceRuleTriggers"] = trigger_lists

# print("\n=== 변동성룰 결과(불량행 샘플) ===")
# print(df_mean_tr[df_mean_tr['ClusterLabel'] > 0][
#    ['Date','Lot','ClusterLabel','VarianceRuleFlag','VarianceRuleHits','VarianceRuleMaxMargin','VarianceRuleTriggers']
# ].head())

# if excluded_vars:
#    excl_df = pd.DataFrame(excluded_vars, columns=['variable','reason']).drop_duplicates()
#    print("\n[경고] 제외/오류 변수 목록:")
#    print(excl_df.head(20))

# ============================================
# D) 성능 평가(이진: 정상 0 vs 불량 1)
# ============================================
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

y_true = (df_mean_tr["ClusterLabel"] > 0).astype(int)
y_pred = df_mean_tr["VarianceRuleFlag"].astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

# print("\n=== Confusion Matrix (0:정상, 1:불량) ===")
# print(pd.DataFrame(cm, index=['True_0','True_1'], columns=['Pred_0','Pred_1']))
# print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
# print(f"Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

# 정상군 오탐률
if (df_mean_tr["ClusterLabel"] == 0).any():
    fpr_normal = (
        df_mean_tr.loc[df_mean_tr["ClusterLabel"] == 0, "VarianceRuleFlag"] == 1
    ).mean()
#    print(f"False Positive Rate on Normal: {fpr_normal:.3%}")

# ============================================
# E) 불량 유형별 탐지율/트리거 빈도 리포트
# ============================================
# 불량군 리콜
per_cluster = []
for c in sorted(df_mean_tr["ClusterLabel"].unique()):
    if c == 0:
        continue
    m = df_mean_tr["ClusterLabel"] == c
    if m.sum() == 0:
        continue
    per_cluster.append(
        {
            "ClusterLabel": c,
            "Support": int(m.sum()),
            "Recall": float((df_mean_tr.loc[m, "VarianceRuleFlag"] == 1).mean()),
        }
    )
# if per_cluster:
#    print("\n=== Per-Cluster Recall ===")
#    print(pd.DataFrame(per_cluster))

# 트리거 변수 Top-N
if "VarianceRuleTriggers" in df_mean_tr.columns:
    trig_series = (
        df_mean_tr.loc[df_mean_tr["VarianceRuleFlag"] == 1, "VarianceRuleTriggers"]
        .str.split(",", expand=True)
        .stack()
        .str.strip()
    )
    trig_series = trig_series[trig_series != ""]
    trigger_counts = trig_series.value_counts().reset_index()
    trigger_counts.columns = ["변수", "TriggerCount"]
#    print("\n=== Top Trigger Variables ===")
#    print(trigger_counts.head(20))

# ============================================
# F) 참고: 클러스터별 적용 변수·임계값 표 (검수용)
# ============================================
df_cluster_thresholds = (
    pd.concat(
        [g.assign(ClusterLabel=c) for c, g in cluster_thr_df_map.items()],
        ignore_index=True,
    )[["ClusterLabel", "변수", "임계값_tr", "direction"]]
    .sort_values(["ClusterLabel", "변수"])
    .reset_index(drop=True)
)
# print("\n=== 클러스터별 적용 변수·임계값(변환 스케일) ===")
# print(df_cluster_thresholds.head(30))
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

# 실제 라벨: ClusterLabel (0,1,2,3)
y_true = df_mean_tr["ClusterLabel"].astype(int)

# 예측 라벨 만들기 (예: 정상은 0, 불량이면 실제 라벨 그대로 예측했다고 가정)
y_pred = df_mean_tr.apply(
    lambda r: r["ClusterLabel"] if r["VarianceRuleFlag"] == 1 else 0, axis=1
).astype(int)

# 혼동 행렬 확인
labels = sorted(y_true.unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(
    cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels]
)
# print("=== Confusion Matrix (multi-class) ===")
# print(cm_df)

# 다중 클래스 Precision, Recall, F1 계산 (클래스별)
prec, rec, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, zero_division=0
)

metrics_df = pd.DataFrame(
    {"Class": labels, "Support": support, "Precision": prec, "Recall": rec, "F1": f1}
)

# === variance 결과 저장(LOT 단위) ===
df_mean_tr["PredCluster"] = y_pred  # 0/1/2/3
out_cols = ["Date", "Lot", "PredCluster", "VarianceRuleTriggers"]  # ← 트리거 포함
df_mean_tr[out_cols].to_csv("pred_variance.csv", index=False)


# print("\n=== Per-Class Metrics ===")
# print(metrics_df)

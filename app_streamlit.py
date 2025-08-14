# app.py
# 실행: streamlit run app.py

from pathlib import Path
import os, json, joblib, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===== 페이지 설정 + 좌우 패딩 제거 =====
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      .main .block-container {
        max-width: 100%;
        padding-left: 0rem;
        padding-right: 0rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
# =======================================

# -------- 경로 설정(필요 시만 수정) --------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
LGBM_PATH = MODELS_DIR / "lgbm.pkl"  # 분류 모델 경로
VR_PATH = MODELS_DIR / "variance.pkl"  # 변동성룰 번들 경로
FEATS_PATH = MODELS_DIR / "selected_features_top15.json"

# 원본 시계열 데이터(csv)
DATA_PATH = "/Users/t2023-m0056/Downloads/df_mean.csv"
# 분류 입력 데이터(csv; 모델 입력용)
DF_FINAL_PATH = "/Users/t2023-m0056/Downloads/df_final_vif.csv"
# 타임스탬프 매핑 데이터(csv; DateTime 포함)
DF_IFFF_PATH = "/Users/t2023-m0056/Downloads/df_ifff.csv"
# -----------------------------------------


# ===== 유틸 =====
@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 날짜 후보 일괄 파싱
    for c in df.columns:
        if any(
            k in c.lower()
            for k in ["time", "date", "datetime", "일자", "날짜", "시각", "timestamp"]
        ):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_resource
def load_models(lgbm_path: Path, vr_path: Path):
    loaded_clf = joblib.load(lgbm_path)
    clf = (
        loaded_clf["model"]
        if isinstance(loaded_clf, dict) and "model" in loaded_clf
        else loaded_clf
    )
    vr_bundle = joblib.load(vr_path)
    return clf, loaded_clf, vr_bundle


def get_feature_list(loaded_clf, clf, feats_path: Path):
    # 1) json 파일 우선
    if feats_path.exists():
        try:
            return json.loads(feats_path.read_text())
        except Exception:
            pass
    # 2) 번들 내 보관 피처
    if isinstance(loaded_clf, dict) and "features" in loaded_clf:
        return loaded_clf["features"]
    # 3) LightGBM 모델에서 추출
    try:
        return list(getattr(clf.booster_, "feature_name")())
    except Exception:
        return []


def _transform_series_by_tuple(s: pd.Series, transformer):
    """variance.pkl의 (kind, obj) 규약 기반 변환. NaN 안전 처리."""
    kind, obj = transformer
    s = pd.to_numeric(s, errors="coerce")
    if kind in ("standard", "robust"):
        mask = s.notna()
        out = pd.Series(np.nan, index=s.index, dtype=float)
        if mask.any():
            out.loc[mask] = obj.transform(s.loc[mask].values.reshape(-1, 1)).ravel()
        return out
    if kind == "log":
        return np.log1p(s + obj["eps"])
    if kind == "log_shift":
        return np.log1p(s + obj["shift"] + obj["eps"])
    return s


def over_mask_for_var(
    df_in: pd.DataFrame, var: str, cluster_col: str, vr_bundle
) -> pd.Series:
    """변동성룰 마스크 계산: 동일 스케일에서 임계 비교 → 원본 위 표시"""
    tr = vr_bundle["transformers"].get(var)
    if tr is None or var not in df_in.columns:
        return pd.Series(False, index=df_in.index)

    s_tr = _transform_series_by_tuple(pd.to_numeric(df_in[var], errors="coerce"), tr)

    thr_map = vr_bundle[
        "cluster_thr_df_map"
    ]  # {cluster: DataFrame[변수, 임계값_tr, direction]}
    thr_s = pd.Series(np.nan, index=df_in.index, dtype=float)
    dir_s = pd.Series(np.nan, index=df_in.index, dtype=float)

    # 각 클러스터별 해당 변수 임계 복사
    if cluster_col not in df_in.columns:
        df_in = df_in.copy()
        df_in[cluster_col] = 0

    for c, tdf in thr_map.items():
        sub = tdf[tdf["변수"] == var]
        if sub.empty:
            continue
        thr = (
            float(sub["임계값_tr"].iloc[0])
            if pd.notna(sub["임계값_tr"].iloc[0])
            else np.nan
        )
        dct = int(sub["direction"].iloc[0])
        idx = df_in.index[df_in[cluster_col].astype(int) == int(c)]
        thr_s.loc[idx] = thr
        dir_s.loc[idx] = dct

    return ((dir_s * (s_tr - thr_s)) > 0).fillna(False)


# ===== 데이터 로드 =====
if not Path(DATA_PATH).exists():
    st.error(f"데이터 파일 없음: {DATA_PATH}")
    st.stop()
df = load_df(DATA_PATH)

# 날짜 컬럼 탐색
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
if not date_cols:
    st.error("날짜 컬럼 없음. 'date/일자/날짜/time' 포함 컬럼 필요.")
    st.stop()
date_col = date_cols[0]

# LOT 컬럼 탐색
lot_cols = [c for c in df.columns if "lot" in c.lower()]
if not lot_cols:
    st.error("LOT 컬럼 없음. 'lot' 포함 컬럼 필요.")
    st.stop()
lot_col = lot_cols[0]

# ===== 상단 컨트롤 =====
c1, c2, _spacer = st.columns([1, 1, 2], gap="small")
with c1:
    time_mode = st.radio("기간", ["전체", "기간 선택"], index=0, horizontal=True)
with c2:
    view_set = st.radio(
        "보기", ["전체", "결함1", "결함2", "결함3"], index=0, horizontal=True
    )

# 날짜 범위 자동화
date_range = None
if time_mode == "기간 선택":
    col_min = pd.to_datetime(df[date_col]).min().date()
    col_max = pd.to_datetime(df[date_col]).max().date()
    date_range = st.date_input(
        "날짜 범위", [col_min, col_max], min_value=col_min, max_value=col_max
    )

# 보기 변수 세트
VARSETS = {
    "전체": ["pH_std6", "Resistance_std6", "Resistance_ma6", "Power_std6"],
    "결함1": ["pH_std6", "Resistance_std6"],
    "결함2": ["Resistance_ma6"],
    "결함3": ["pH_std6", "Power_std6"],
}
present = [c for c in VARSETS[view_set] if c in df.columns]
if not present:
    st.error(f"y축 변수 없음: {VARSETS[view_set]}")
    st.stop()

# 수치화 + 시간 정렬
df_plot = df.copy()
for c in present:
    df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
df_plot = df_plot.sort_values(date_col)

st.title("Line Chart")

# 기간 필터 적용
if time_mode == "기간 선택" and date_range and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df_plot = df_plot[(df_plot[date_col] >= start) & (df_plot[date_col] < end)]

# ===== 모델 로드 =====
missing = [p for p in [LGBM_PATH, VR_PATH] if not Path(p).exists()]
if missing:
    st.error(
        "모델 파일 없음:\n"
        + "\n".join(str(p) for p in missing)
        + f"\n\n확인: MODELS_DIR={MODELS_DIR}"
    )
    st.stop()

clf, loaded_clf, vr_bundle = load_models(LGBM_PATH, VR_PATH)

# ===== 분류 예측 =====
if not Path(DF_FINAL_PATH).exists():
    st.error(f"분류 입력 파일 없음: {DF_FINAL_PATH}")
    st.stop()
df_clf = pd.read_csv(DF_FINAL_PATH)

# 날짜/LOT 동기화
if date_col in df_clf.columns:
    df_clf[date_col] = pd.to_datetime(df_clf[date_col], errors="coerce")
else:
    st.error(f"df_final_vif에 날짜 컬럼({date_col}) 없음. 컬럼명 확인 필요.")
    st.stop()
if lot_col not in df_clf.columns:
    st.error(f"df_final_vif에 LOT 컬럼({lot_col}) 없음. 컬럼명 확인 필요.")
    st.stop()

# === [추가] LOT 키 타입 통일 ===
df_plot[lot_col] = df_plot[lot_col].astype(str)
df_clf[lot_col] = df_clf[lot_col].astype(str)

# === [추가] 예측 대상 원본 키로 제한(증강 제거) ===
orig_keys = df_plot[[date_col, lot_col]].dropna().drop_duplicates()
df_clf = df_clf.merge(orig_keys, on=[date_col, lot_col], how="inner")

# === [수정] 모델 피처 목록 고정 + 누락 피처 보강 ===
top15 = get_feature_list(loaded_clf, clf, Path(FEATS_PATH))
if not top15:
    st.error("모델 피처 목록 비어있음. lgbm.pkl 생성 로직 확인 필요.")
    st.stop()
missing_feats = [f for f in top15 if f not in df_clf.columns]
for f in missing_feats:
    df_clf[f] = np.nan
Xc = df_clf[top15].apply(pd.to_numeric, errors="coerce")
Xc = Xc.fillna(Xc.median(numeric_only=True))
df_clf["pred_label"] = pd.Series(clf.predict(Xc)).astype(int)


# (날짜, LOT)별 최신값으로 축약 후 병합
pred_last = df_clf.sort_values(date_col).drop_duplicates(
    subset=[date_col, lot_col], keep="last"
)
df_plot = df_plot.merge(
    pred_last[[date_col, lot_col, "pred_label"]], on=[date_col, lot_col], how="left"
)
df_plot["pred_label"] = df_plot["pred_label"].fillna(0).astype(int)


# ===== 변동성룰 마스크 =====
if "ClusterLabel" not in df_plot.columns:
    df_plot["ClusterLabel"] = 0
CLUSTER_COL = "ClusterLabel" if "ClusterLabel" in df_plot.columns else "pred_label"

for y in present:
    df_plot[f"over_{y}"] = over_mask_for_var(df_plot, y, CLUSTER_COL, vr_bundle)

# ===== df_ifff(DateTime) 병합 =====
if not Path(DF_IFFF_PATH).exists():
    st.error(f"타임스탬프 파일 없음: {DF_IFFF_PATH}")
    st.stop()
df_ifff = pd.read_csv(DF_IFFF_PATH)

if "DateTime" not in df_ifff.columns:
    st.error("df_ifff에 'DateTime' 컬럼 없음")
    st.stop()
df_ifff["DateTime"] = pd.to_datetime(df_ifff["DateTime"], errors="coerce")

# 키 컬럼 동기화
if date_col not in df_ifff.columns:
    df_ifff[date_col] = df_ifff["DateTime"].dt.normalize()
else:
    df_ifff[date_col] = pd.to_datetime(df_ifff[date_col], errors="coerce")

# === [추가] LOT 키 타입 통일 ===
df_ifff[lot_col] = df_ifff[lot_col].astype(str)

if lot_col not in df_ifff.columns:
    st.error(f"df_ifff에 LOT 컬럼({lot_col}) 없음")
    st.stop()

# (날짜, LOT)별 최신 DateTime 매핑 → 병합
ts_map = df_ifff.sort_values("DateTime").drop_duplicates(
    subset=[date_col, lot_col], keep="last"
)[[date_col, lot_col, "DateTime"]]
df_plot = df_plot.merge(ts_map, on=[date_col, lot_col], how="left")

# ===== Plotly 멀티 그래프 + 선택 연동 =====
_plot_key = "plt_main"
_prev = st.session_state.get(_plot_key)
_has_sel = bool(
    _prev and getattr(_prev, "selection", None) and _prev.selection.get("points")
)

if _has_sel:
    left_col, right_col = st.columns([5, 2], gap="small")
else:
    left_col = st.container()
    right_col = None

rows = len(present)
fig = make_subplots(
    rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=present
)

# 색상 팔레트
palette = ["#6dc2ff", "#64dcc4", "#ffe588", "#91c2ff"]
color_map = {y: palette[i % len(palette)] for i, y in enumerate(present)}

for i, y in enumerate(present, start=1):
    d = df_plot[["DateTime", y, lot_col, "pred_label", f"over_{y}"]].copy()
    d = d[pd.notna(d["DateTime"])]

    # 평균 라인
    mean_df = d.groupby("DateTime", as_index=False)[y].mean()
    fig.add_trace(
        go.Scatter(
            x=mean_df["DateTime"],
            y=mean_df[y],
            mode="lines",
            name=f"{y} mean",
            line=dict(width=2, color=color_map[y]),
            hovertemplate=f"DateTime=%{{x}}<br>{y}(mean)=%{{y:.3f}}<extra></extra>",
        ),
        row=i,
        col=1,
    )

    # 모든 점(클릭 타깃)
    custom = np.stack(
        [
            d[lot_col].astype(str).values,
            d["DateTime"].astype(str).values,
            np.full(len(d), y),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=d["DateTime"],
            y=d[y],
            mode="markers",
            name=f"{y} points",
            marker=dict(size=4, color=color_map[y], line=dict(width=1, color="black")),
            selected=dict(marker=dict(size=14, color=color_map[y], opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.25)),
            customdata=custom,
            hovertemplate=f"{lot_col}=%{{customdata[0]}}<br>DateTime=%{{x}}<br>{y}=%{{y}}<extra></extra>",
        ),
        row=i,
        col=1,
    )

    # 변동성룰 초과(빨간 원)
    od = d[d[f"over_{y}"]]
    if not od.empty:
        od_custom = np.stack(
            [
                od[lot_col].astype(str).values,
                od["DateTime"].astype(str).values,
                np.full(len(od), y),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=od["DateTime"],
                y=od[y],
                mode="markers",
                name=f"over {y}",
                marker=dict(size=11, color="red", line=dict(width=1, color="red")),
                customdata=od_custom,
                hovertemplate=f"OVER<br>{lot_col}=%{{customdata[0]}}<br>DateTime=%{{x}}<br>{y}=%{{y}}<extra></extra>",
            ),
            row=i,
            col=1,
        )

    # 분류 불량(검정 X) — (DateTime, LOT) 단위 1회 표시
    xd = d[d["pred_label"] > 0].copy()
    if not xd.empty:
        # 같은 (DateTime, LOT) 키에서 하나만 남김
        xd = xd.sort_values(["DateTime", lot_col]).drop_duplicates(
            subset=["DateTime", lot_col], keep="last"
        )
        # [선택] 첫 번째 패널(= i==1)에만 X 표시해 과밀 방지
        if i == 1:
            fig.add_trace(
                go.Scatter(
                    x=xd["DateTime"],
                    y=xd[y],  # 해당 패널의 y값 위치에 X
                    mode="text",
                    text="×",
                    textfont=dict(size=23, color="black"),
                    name="pred defect",
                    hoverinfo="skip",
                ),
                row=i,
                col=1,
            )


fig.update_layout(
    margin=dict(l=0, r=0, t=30, b=0),
    height=max(260 * rows, 320),
    showlegend=False,
    clickmode="event+select",
)

with left_col:
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=_plot_key,
    )

# 우측 패널
if right_col:
    with right_col:
        st.header("LOT 상세")
        st.markdown(
            """
            <style>
              .lot-box {
                background-color: #E6FFF6;
                border: 1px solid #B9F5E6;
                border-radius: 12px;
                padding: 14px 16px;
              }
              .lot-item { margin: 0 0 8px 0; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        sel = event if isinstance(event, dict) else _prev
        pts = (sel or {}).get("selection", {}).get("points", []) if sel else []
        pt = next(
            (p for p in pts if "customdata" in p and len(p["customdata"]) >= 3), None
        )

        if pt:
            sel_lot = str(pt["customdata"][0])
            sel_dt = pd.to_datetime(pt.get("x") or pt["customdata"][1])
            sel_metric = pt["customdata"][2]
            sel_y_val = pt.get("y", None)

            lot_rows = df_plot[df_plot[lot_col].astype(str) == sel_lot]
            if "pred_label" in lot_rows.columns and not lot_rows.empty:
                _pred_col = pd.to_numeric(lot_rows["pred_label"], errors="coerce")
                pred = int(_pred_col.dropna().max()) if _pred_col.notna().any() else 0
            else:
                pred = 0

            status_map = {0: "정상", 1: "불량1", 2: "불량2", 3: "불량3"}
            defect_name_map = {
                1: "색상 불량 / 흑색 불량",
                2: "도금층 불균일",
                3: "수소취성",
            }
            cause_map = {
                1: "- 산처리 불충분으로 표면 산화물 잔류 → 도금 후 색상 불균일",
                2: "전류밀도/전압 편차 → 미세 균열·핀홀·산화물 침착",
                3: "- pH 과변동/전력 변화 → 수소 발생 증가 → 수소취성",
            }
            procvar_map = {
                1: "- pH 이동표준편차 ↑, 저항 이동표준편차 ↑",
                2: "- 저항 이동평균 ↑",
                3: "- pH 이동표준편차 ↑, 전력 이동평균 ↑",
            }

            if time_mode == "기간 선택" and date_range and len(date_range) == 2:
                start_dt = pd.to_datetime(date_range[0])
                end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            else:
                start_dt = pd.to_datetime(df_ifff["DateTime"]).min()
                end_dt = pd.to_datetime(df_ifff["DateTime"]).max()

            lot_ifff = df_ifff[
                (df_ifff[lot_col].astype(str) == sel_lot)
                & (df_ifff["DateTime"] >= start_dt)
                & (df_ifff["DateTime"] < end_dt)
            ]
            dt_first = lot_ifff["DateTime"].min()
            dt_last = lot_ifff["DateTime"].max()

            cols_avg = ["pH", "Temp", "Current", "Voltage", "Resistance", "Power"]
            avail = [c for c in cols_avg if c in lot_ifff.columns]
            means = {
                c: pd.to_numeric(lot_ifff[c], errors="coerce").mean() for c in avail
            }

            def fmt_means(d):
                order = ["pH", "Temp", "Current", "Voltage", "Resistance", "Power"]
                out = []
                for k in order:
                    v = d.get(k, None)
                    if v is not None and pd.notna(v):
                        out.append(f"{k}={v:.3f}")
                return ", ".join(out) if out else "-"

            start_s = dt_first.strftime("%Y-%m-%d %H:%M") if pd.notna(dt_first) else "-"
            end_s = dt_last.strftime("%Y-%m-%d %H:%M") if pd.notna(dt_last) else "-"

            st.markdown(
                f"""
                <div class="lot-box">
                  <div class="lot-item">✅ 상태 :  {status_map.get(pred, "정상")}</div>
                  <div class="lot-item">✅ 기간 :  {start_s} ~ {end_s}</div>
                  <div class="lot-item">✅ LOT :  {sel_lot}</div>
                  <div class="lot-item">✅ 불량유형 :  {defect_name_map.get(pred, "-")}</div>
                  <div class="lot-item">✅ 결함원인 :  {cause_map.get(pred, "-")}</div>
                  <div class="lot-item">✅ 공정 파라미터 평균 :  {fmt_means(means)}</div>
                  <div class="lot-item">✅ 원인변수 :  {sel_metric} = {sel_y_val if sel_y_val is not None else "-"}</div>
                  <div class="lot-item">✅ 공정 변수 :  {procvar_map.get(pred, "-")}</div>
                  <div class="lot-item">✅ 공정최적화방안 :  -</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

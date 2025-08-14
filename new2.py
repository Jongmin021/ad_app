# overlay_app.py
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date

# --- ⚙️ 파일 경로 (상대 경로 사용 권장) ---
PATH_RAW_SENSOR = "/Users/t2023-m0056/Desktop/파일/df_sensor.csv"
PATH_THRESHOLDS = "/Users/t2023-m0056/Desktop/파일/df_thresh_z (1).csv"
PATH_GROUND_TRUTH = "ground_truth_for_app.csv"
PATH_PRED_LGBM = "pred_lot.csv"
PATH_PRED_VARIANCE = "pred_variance.csv"

st.set_page_config(
    page_title="Sensor Overlay Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# [수정] 대시보드 제목
st.title("📊 Sensor Overlay Dashboard")

st.markdown(
    """<style>
[data-testid="stAppViewContainer"] .main .block-container {padding:1rem 1rem 0rem 1rem;}
[data-testid="stHeader"]{display:none;} footer{visibility:hidden;}
</style>""",
    unsafe_allow_html=True,
)


# --- 데이터 로딩 함수들 ---
@st.cache_data
def load_raw():
    df = pd.read_csv(PATH_RAW_SENSOR)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    return df.dropna(subset=["DateTime"]).sort_values("DateTime")


@st.cache_data
def load_thresholds():
    df = pd.read_csv(PATH_THRESHOLDS)
    df["ClusterLabel"] = pd.to_numeric(df["ClusterLabel"], errors="coerce").astype(
        "Int64"
    )
    df["임계값"] = pd.to_numeric(df["임계값"], errors="coerce")
    return df


@st.cache_data
def load_preds():
    frames = []
    try:
        a = pd.read_csv(PATH_PRED_LGBM)
        a["Source"] = "lgbm"
        a["PredCluster"] = (
            pd.to_numeric(a["PredCluster"], errors="coerce").fillna(0).astype(int)
        )
        a["VarianceRuleTriggers"] = ""
        frames.append(a)
    except FileNotFoundError:
        st.warning(
            f"{PATH_PRED_LGBM} 파일을 찾을 수 없습니다. lgbm.py를 먼저 실행하여 예측 파일을 생성해주세요."
        )
    try:
        b = pd.read_csv(PATH_PRED_VARIANCE)
        b["Source"] = "variance"
        b["PredCluster"] = (
            pd.to_numeric(b["PredCluster"], errors="coerce").fillna(0).astype(int)
        )
        if "VarianceRuleTriggers" not in b.columns:
            b["VarianceRuleTriggers"] = ""
        frames.append(b)
    except FileNotFoundError:
        st.warning(
            f"{PATH_PRED_VARIANCE} 파일을 찾을 수 없습니다. variance.py를 실행하여 예측 파일을 생성해주세요."
        )

    if not frames:
        return pd.DataFrame(
            columns=["Date", "Lot", "PredCluster", "VarianceRuleTriggers", "Source"]
        )

    pred_all = pd.concat(frames, ignore_index=True)
    pred_all["TriggersList"] = (
        pred_all["VarianceRuleTriggers"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .apply(lambda L: [s.strip() for s in L if s and s.strip()])
    )
    return pred_all


@st.cache_data
def load_ground_truth():
    try:
        df = pd.read_csv(PATH_GROUND_TRUTH)
        return df
    except FileNotFoundError:
        st.error(
            f"정답 파일({PATH_GROUND_TRUTH})을 찾을 수 없습니다. lgbm.py를 먼저 실행하여 파일을 생성해주세요."
        )
        return pd.DataFrame(columns=["Date", "Lot", "ClusterLabel"])


# --- 데이터 로드 및 전처리 ---
raw = load_raw()
pred = load_preds()
thr = load_thresholds()
truth = load_ground_truth()

# 키 정규화
raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce").dt.date
pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce").dt.date
truth["Date"] = pd.to_datetime(truth["Date"], errors="coerce").dt.date
raw["Lot"] = raw["Lot"].astype(str)
pred["Lot"] = pred["Lot"].astype(str)
truth["Lot"] = truth["Lot"].astype(str)

lot_span = (
    raw.groupby(["Date", "Lot"])["DateTime"].agg(start="min", end="max").reset_index()
)

# --- UI 영역 ---
c1, c2 = st.columns([1, 1])
with c1:
    mode = st.selectbox("기간 모드", ["전체", "기간선택"], index=0, key="date_mode")
    limit_start, limit_end = raw["DateTime"].min().date(), raw["DateTime"].max().date()

    if mode == "기간선택":
        sel_range = st.date_input(
            "기간 선택",
            value=(limit_start, limit_end),
            min_value=limit_start,
            max_value=limit_end,
            format="YYYY-MM-DD",
        )
        try:
            start_d, end_d = sel_range
        except ValueError:
            start_d, end_d = limit_start, limit_end
    else:
        start_d, end_d = limit_start, limit_end

with c2:
    kind = st.selectbox(
        "불량 유형", ["전체", "불량 유형 1", "불량 유형 2", "불량 유형 3"], index=0
    )

cluster_map = {
    "전체": [1, 2, 3],
    "불량 유형 1": [1],
    "불량 유형 2": [2],
    "불량 유형 3": [3],
}
clusters = cluster_map[kind]
thr_cluster = clusters[0] if len(clusters) == 1 else None

# --- 데이터 필터링 ---
date_filter = (raw["DateTime"].dt.date >= start_d) & (raw["DateTime"].dt.date <= end_d)
base = raw[date_filter].copy()

pred_filtered = pred[pred["Date"].between(start_d, end_d)]
bad_span = lot_span.merge(pred_filtered, on=["Date", "Lot"], how="inner")
bad_span = bad_span[bad_span["PredCluster"].isin(clusters)]
bad_span_var = bad_span[bad_span["Source"] == "variance"].copy()
bad_span_lgbm = bad_span[bad_span["Source"] == "lgbm"].copy()

# --- 상태 관리 ---
if "selected_point" not in st.session_state:
    st.session_state["selected_point"] = None


def clear_selection():
    st.session_state["selected_point"] = None


plot_cols_map = {
    "전체": ["pH_std6", "Resistance_std6", "Resistance_ma6", "Power_std6"],
    "불량 유형 1": ["pH_std6", "Resistance_std6"],
    "불량 유형 2": ["Resistance_ma6"],
    "불량 유형 3": ["pH_std6", "Power_std6"],
}
ycols = plot_cols_map[kind]

# --- 레이아웃 ---
left, right = st.columns([5, 2], gap="small")

with left:
    if base.empty:
        st.warning("선택 구간에 데이터가 없습니다.")
    else:
        base["LotNum"] = pd.to_numeric(base["Lot"], errors="coerce")
        base = base.dropna(subset=["LotNum"])
        base["LotNum"] = base["LotNum"].astype(int)
        base = base[(base["LotNum"] >= 1) & (base["LotNum"] <= 22)]

        span = (
            base.groupby(["Date", "Lot"], as_index=False)["DateTime"]
            .agg(start="min", end="max")
            .assign(mid=lambda d: d["start"] + (d["end"] - d["start"]) / 2)
        )

        for col in ycols:
            agg_val = (
                base.groupby(["Date", "Lot"], as_index=False)[col]
                .mean()
                .rename(columns={col: "Value"})
            )
            agg = agg_val.merge(span, on=["Date", "Lot"], how="left").sort_values("mid")

            # [수정] 라인 그래프 생성 (x="mid" 사용)
            fig = px.line(agg, x="mid", y="Value", title=col)

            for _, r in bad_span.iterrows():
                fig.add_vrect(
                    x0=r["start"],
                    x1=r["end"],
                    fillcolor="red",
                    opacity=0.1,
                    line_width=0,
                )

            # LGBM 예측 점 추가
            if not bad_span_lgbm.empty:
                pts_lgbm = bad_span_lgbm.merge(agg, on=["Date", "Lot"], how="inner")
                if not pts_lgbm.empty:
                    pts_lgbm["Triggers"] = ""
                    fig.add_scatter(
                        x=pts_lgbm["mid"],
                        y=pts_lgbm["Value"],
                        mode="markers",
                        name="LGBM Prediction",
                        marker=dict(
                            size=12, symbol="x", color="red", line=dict(width=2)
                        ),
                        customdata=pts_lgbm[
                            [
                                "Date",
                                "Lot",
                                "PredCluster",
                                "Source",
                                "Triggers",
                                "Value",
                                "mid",
                            ]
                        ]
                        .astype(str)
                        .values,
                        hovertemplate="<b>%{customdata[3]} Prediction</b><br>"
                        + "Date: %{customdata[0]}<br>"
                        + "Lot: %{customdata[1]}<br>"
                        + "Value: %{customdata[5]}<extra></extra>",
                    )

            # Variance 예측 점 추가
            if not bad_span_var.empty:
                var_lot = bad_span_var[
                    bad_span_var["TriggersList"].apply(lambda L: col in L)
                ]
                pts_var = var_lot.merge(agg, on=["Date", "Lot"], how="inner")
                if not pts_var.empty:
                    pts_var = pts_var.rename(
                        columns={"VarianceRuleTriggers": "Triggers"}
                    )
                    fig.add_scatter(
                        x=pts_var["mid"],
                        y=pts_var["Value"],
                        mode="markers",
                        name="Variance Prediction",
                        marker=dict(
                            size=12, symbol="circle", color="blue", line=dict(width=2)
                        ),
                        customdata=pts_var[
                            [
                                "Date",
                                "Lot",
                                "PredCluster",
                                "Source",
                                "Triggers",
                                "Value",
                                "mid",
                            ]
                        ]
                        .astype(str)
                        .values,
                        hovertemplate="<b>%{customdata[3]} Prediction</b><br>"
                        + "Date: %{customdata[0]}<br>"
                        + "Lot: %{customdata[1]}<br>"
                        + "Triggers: %{customdata[4]}<br>"
                        + "Value: %{customdata[5]}<extra></extra>",
                    )

            if thr_cluster is not None:
                t = thr[(thr["ClusterLabel"] == thr_cluster) & (thr["변수"] == col)]
                if not t.empty and pd.notna(t.iloc[0]["임계값"]):
                    fig.add_hline(
                        y=float(t.iloc[0]["임계값"]), line_dash="dot", line_color="grey"
                    )

            # [수정] 레이아웃 업데이트
            fig.update_layout(
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis_title=None,
                yaxis_title=col,
                hovermode="x unified",
                height=300,
            )

            # [수정] x축 범위 설정
            # 사용자가 선택한 기간에 맞춰 x축 범위를 동적으로 설정
            x_range_start = pd.to_datetime(start_d)
            x_range_end = pd.to_datetime(end_d) + pd.Timedelta(days=1)
            fig.update_xaxes(range=[x_range_start, x_range_end], fixedrange=True)

            if not agg.empty:
                pad = (agg["Value"].max() - agg["Value"].min()) * 0.1
                fig.update_yaxes(
                    range=[agg["Value"].min() - pad, agg["Value"].max() + pad]
                )

            # [수정] st.plotly_chart 함수
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": "reset",
                },
            )

with right:
    st.markdown("#### LOT 상세 정보")
    sel = st.session_state.get("selected_point")
    if not sel:
        st.info("차트의 점(point)을 클릭하면 상세 정보가 표시됩니다.")
    else:
        for key, value in sel.items():
            st.markdown(f"- **{key}**: {value}")

        if "Date" in sel and "Lot" in sel:
            sel_date_dt = pd.to_datetime(sel.get("Date")).date()
            info = lot_span[
                (lot_span["Date"] == sel_date_dt) & (lot_span["Lot"] == sel.get("Lot"))
            ]
            if not info.empty:
                st.markdown(
                    f"- **LOT 구간**: {info.iloc[0]['start']} ~ {info.iloc[0]['end']}"
                )

        st.button("닫기", use_container_width=True, on_click=clear_selection)

# --- 성능 평가 섹션 ---
st.divider()
st.markdown("### 📊 모델 예측 성능 종합")


def display_performance_pie(pred_df, truth_df, title, colors):
    if truth_df.empty:
        return

    pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.date
    truth_df["Date"] = pd.to_datetime(truth_df["Date"]).dt.date
    pred_df["Lot"] = pred_df["Lot"].astype(str)
    truth_df["Lot"] = truth_df["Lot"].astype(str)

    perf_df = pd.merge(pred_df, truth_df, on=["Date", "Lot"], how="inner")

    if perf_df.empty:
        st.warning(f"{title} 모델이 평가할 원본 LOT 데이터를 찾을 수 없습니다.")
        return

    total_lots = len(perf_df)
    perf_df["Correct"] = perf_df["PredCluster"] == perf_df["ClusterLabel"]
    correct_count = perf_df["Correct"].sum()
    accuracy = (correct_count / total_lots) * 100 if total_lots > 0 else 0
    counts = (
        perf_df["Correct"].value_counts().rename({True: "Correct", False: "Incorrect"})
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"**{title} Accuracy**", value=f"{accuracy:.2f} %")
    with col2:
        st.metric(label="**Evaluated LOTs**", value=f"{total_lots} 개")

    fig_perf = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.4,
        color=counts.index,
        color_discrete_map=colors,
        title=f"{title} 모델 성능",
    )
    fig_perf.update_traces(
        textposition="outside",
        textinfo="percent+label",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig_perf.update_layout(
        showlegend=False, margin=dict(t=40, b=20, l=20, r=20), height=300
    )
    st.plotly_chart(fig_perf, use_container_width=True)


c3, c4 = st.columns(2)
with c3:
    display_performance_pie(
        pred[pred["Source"] == "lgbm"].copy(),
        truth.copy(),
        "LGBM",
        {"Correct": "#1f77b4", "Incorrect": "#ff7f0e"},
    )
with c4:
    display_performance_pie(
        pred[pred["Source"] == "variance"].copy(),
        truth.copy(),
        "Variance",
        {"Correct": "#2ca02c", "Incorrect": "#d62728"},
    )

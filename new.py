# overlay_app.py
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date

st.set_page_config(
    page_title="Sensor Overlay Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 대시보드 제목 (변경)
st.title("품질예측 대시보드")

st.markdown(
    """<style>
/* 레이아웃 여백/헤더 숨김 */
[data-testid="stAppViewContainer"] .main .block-container {padding:0rem;}
[data-testid="stHeader"]{display:none;} footer{visibility:hidden;}

/* 기본 라벨/값 규칙 (그대로 유지) */
.lbl {font-size:20px; font-weight:700;}
.val {font-size:18px; font-weight:400;}

/* ───── Expander(토글) 스타일 ───── */
/* 제목(요약) — 더 크게/두껍게 */
div[data-testid="stExpander"] details summary {
  font-size: 26px !important;
  font-weight: 1000 !important;
  line-height: 1.3 !important;
}

/* 내용(펼쳐진 영역) — 더 작게 */
div[data-testid="stExpander"] > div[role="region"] {
  font-size: 16px !important;
  line-height: 1.9 !important;
}
</style>""",
    unsafe_allow_html=True,
)


# ------------------------- 데이터 로더 -------------------------
@st.cache_data
def load_raw():
    df = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_sensor.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    return df.dropna(subset=["DateTime"]).sort_values("DateTime")


@st.cache_data
def load_thresholds():
    df = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_thresh_z (1).csv")
    df["ClusterLabel"] = pd.to_numeric(df["ClusterLabel"], errors="coerce").astype(
        "Int64"
    )
    df["임계값"] = pd.to_numeric(df["임계값"], errors="coerce")
    return df


thr = load_thresholds()


@st.cache_data
def load_preds():
    frames = []
    try:
        a = pd.read_csv("pred_lot.csv")
        a["Source"] = "lgbm"
        a["PredCluster"] = (
            pd.to_numeric(a["PredCluster"], errors="coerce").fillna(0).astype(int)
        )
        a["VarianceRuleTriggers"] = ""
        frames.append(a)
    except FileNotFoundError:
        pass
    try:
        b = pd.read_csv("pred_variance.csv")
        b["Source"] = "variance"
        b["PredCluster"] = (
            pd.to_numeric(b["PredCluster"], errors="coerce").fillna(0).astype(int)
        )
        if "VarianceRuleTriggers" not in b.columns:
            b["VarianceRuleTriggers"] = ""
        frames.append(b)
    except FileNotFoundError:
        pass

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


raw = load_raw()
pred = load_preds()

# 키 정규화
raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce").dt.date
pred["Date"] = pd.to_datetime(pred["Date"], errors="coerce").dt.date
raw["Lot"] = raw["Lot"].astype(str)
pred["Lot"] = pred["Lot"].astype(str)


# -------------------- 성능용 정답 라벨 로더 --------------------
@st.cache_data
def load_truth_lot(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "ClusterLabel" in raw_df.columns:
        tru = (
            raw_df.dropna(subset=["ClusterLabel"])
            .assign(ClusterLabel=pd.to_numeric(raw_df["ClusterLabel"], errors="coerce"))
            .dropna(subset=["ClusterLabel"])
        )
        if not tru.empty:
            g = tru.groupby(["Date", "Lot"])["ClusterLabel"].max().reset_index()
            g = g.rename(columns={"ClusterLabel": "TrueCluster"})
            g["Lot"] = g["Lot"].astype(str)
            return g
    try:
        df_mean = pd.read_csv("/Users/t2023-m0056/Desktop/파일/df_mean (1).csv")
        df_mean["Date"] = pd.to_datetime(df_mean["Date"], errors="coerce").dt.date
        df_mean["Lot"] = df_mean["Lot"].astype(str)
        df_mean["ClusterLabel"] = (
            pd.to_numeric(df_mean["ClusterLabel"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        g = df_mean.groupby(["Date", "Lot"])["ClusterLabel"].max().reset_index()
        g = g.rename(columns={"ClusterLabel": "TrueCluster"})
        return g
    except Exception:
        return pd.DataFrame(columns=["Date", "Lot", "TrueCluster"])


truth_lot = load_truth_lot(raw)

# LOT 구간
lot_span = (
    raw.groupby(["Date", "Lot"])["DateTime"].agg(start="min", end="max").reset_index()
)

# --------------------------- UI 상단 ---------------------------
c1, c2 = st.columns([1, 1])
with c1:
    mode = st.selectbox("기간 모드", ["전체", "기간선택"], index=0)
    limit_start, limit_end = raw["DateTime"].min().date(), raw["DateTime"].max().date()

    def normalize_date_range(sel, default_start, default_end):
        if isinstance(sel, (tuple, list)):
            if len(sel) == 2:
                s, e = sel[0], sel[1]
            elif len(sel) == 1:
                s = e = sel[0]
            else:
                s, e = default_start, default_end
        else:
            s = e = sel
        s = pd.to_datetime(s).date()
        e = pd.to_datetime(e).date()
        if s > e:
            s, e = e, s
        return s, e

    if mode == "기간선택":
        sel = st.date_input(
            "기간 선택",
            value=(limit_start, limit_end),
            min_value=limit_start,
            max_value=limit_end,
            format="YYYY-MM-DD",
            key="기간_선택_범위_v2",
        )
        start_d, end_d = normalize_date_range(sel, limit_start, limit_end)
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

# x축 범위
if mode == "기간선택":
    x0 = pd.to_datetime(start_d)
    x1 = pd.to_datetime(end_d) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
else:
    x0 = raw["DateTime"].min()
    x1 = raw["DateTime"].max()

# 예측 결합
bad_span = lot_span.merge(pred, on=["Date", "Lot"], how="inner")
bad_span = bad_span[bad_span["PredCluster"].isin(clusters)]
bad_span_var = bad_span[bad_span["Source"] == "variance"].copy()
bad_span_lgbm = bad_span[bad_span["Source"] == "lgbm"].copy()

# ------------------------ 상태 초기화 ------------------------
if "selected_point" not in st.session_state:
    st.session_state["selected_point"] = None
if "plt_nonce" not in st.session_state:
    st.session_state["plt_nonce"] = 0  # 선택 즉시 해제용

# ---------------- 변수 한글명 ----------------
plot_cols_map = {
    "전체": ["pH_std6", "Resistance_std6", "Resistance_ma6", "Power_std6"],
    "불량 유형 1": ["pH_std6", "Resistance_std6"],
    "불량 유형 2": ["Resistance_ma6"],
    "불량 유형 3": ["pH_std6", "Power_std6"],
}
ycols = plot_cols_map[kind]

kor_var_name = {
    "pH_std6": "pH 이동 표준편차",
    "Resistance_std6": "저항 이동 표준편차",
    "Resistance_ma6": "저항 이동 평균",
    "Power_std6": "전력 이동 표준편차",
}

# ---------------- 상단 그래프 + 우측 상세 ----------------
left, right = st.columns([5, 2], gap="small")

with left:
    base = raw[(raw["DateTime"] >= x0) & (raw["DateTime"] <= x1)].copy()
    base["LotNum"] = pd.to_numeric(base["Lot"], errors="coerce")
    base = base.dropna(subset=["LotNum"])
    base["LotNum"] = base["LotNum"].astype(int)
    base = base[(base["LotNum"] >= 1) & (base["LotNum"] <= 22)]

    if base.empty:
        st.warning("선택 구간에 데이터 없음")
    else:
        span = (
            base.groupby(["Date", "Lot"], as_index=False)["DateTime"]
            .agg(start="min", end="max")
            .assign(mid=lambda d: d["start"] + (d["end"] - d["start"]) / 2)
        )
        span = span.merge(
            base[["Date", "Lot", "LotNum"]].drop_duplicates(),
            on=["Date", "Lot"],
            how="left",
        )

        for col in ycols:
            agg_val = (
                base.groupby(["Date", "Lot"], as_index=False)[col]
                .mean()
                .rename(columns={col: "Value"})
            )
            agg = agg_val.merge(span, on=["Date", "Lot"], how="left").sort_values("mid")

            fig = px.line(agg, x="mid", y="Value", title=kor_var_name.get(col, col))

            # 예측 LOT 밴드
            for _, r in bad_span.iterrows():
                fig.add_vrect(x0=r["start"], x1=r["end"], opacity=0.15, line_width=0)

            # LGBM 포인트
            pts_lgbm = pd.DataFrame()
            if not bad_span_lgbm.empty:
                pts_lgbm = (
                    bad_span_lgbm[["Date", "Lot", "PredCluster"]]
                    .drop_duplicates()
                    .merge(
                        agg[["Date", "Lot", "mid", "Value"]],
                        on=["Date", "Lot"],
                        how="inner",
                    )
                    .sort_values("mid")
                )
                if not pts_lgbm.empty:
                    cd_lgbm = pd.DataFrame(
                        {
                            "Date": pts_lgbm["Date"].astype(str),
                            "Lot": pts_lgbm["Lot"].astype(str),
                            "PredCluster": pts_lgbm["PredCluster"],
                            "Source": "lgbm",
                            "Triggers": "",
                            "Col": col,
                            "Value": pts_lgbm["Value"],
                            "DateTime": pts_lgbm["mid"].astype(str),
                        }
                    )
                    fig.add_scatter(
                        x=pts_lgbm["mid"],
                        y=pts_lgbm["Value"],
                        mode="markers",
                        name="LGBM 점",
                        marker=dict(size=10, symbol="x"),
                        customdata=cd_lgbm.values,
                        hovertemplate="시간=%{x}<br>값=%{y}<br>종류=%{customdata[3]}<extra></extra>",
                    )

            # Variance 포인트
            pts_var = pd.DataFrame()
            if not bad_span_var.empty:
                var_lot = bad_span_var.copy()
                var_lot = var_lot[var_lot["TriggersList"].apply(lambda L: col in L)]
                pts_var = (
                    var_lot[["Date", "Lot", "PredCluster", "VarianceRuleTriggers"]]
                    .drop_duplicates()
                    .merge(
                        agg[["Date", "Lot", "mid", "Value"]],
                        on=["Date", "Lot"],
                        how="inner",
                    )
                    .sort_values("mid")
                )
                if not pts_var.empty:
                    cd_var = pd.DataFrame(
                        {
                            "Date": pts_var["Date"].astype(str),
                            "Lot": pts_var["Lot"].astype(str),
                            "PredCluster": pts_var["PredCluster"],
                            "Source": "variance",
                            "Triggers": pts_var["VarianceRuleTriggers"].fillna(""),
                            "Col": col,
                            "Value": pts_var["Value"],
                            "DateTime": pts_var["mid"].astype(str),
                        }
                    )
                    fig.add_scatter(
                        x=pts_var["mid"],
                        y=pts_var["Value"],
                        mode="markers",
                        name="Variance 점",
                        marker=dict(size=10, symbol="x"),
                        customdata=cd_var.values,
                        hovertemplate="시간=%{x}<br>값=%{y}<br>종류=%{customdata[3]}<extra></extra>",
                    )

            # 통합 히트박스
            hit_df = None
            hit_parts = []
            if not pts_lgbm.empty:
                hit_parts.append(
                    pd.DataFrame(
                        {
                            "x": pts_lgbm["mid"],
                            "y": pts_lgbm["Value"],
                            "Date": pts_lgbm["Date"].astype(str),
                            "Lot": pts_lgbm["Lot"].astype(str),
                            "PredCluster": pts_lgbm["PredCluster"],
                            "Source": "lgbm",
                            "Triggers": "",
                            "Col": col,
                            "Value": pts_lgbm["Value"],
                            "DateTimeStr": pts_lgbm["mid"].astype(str),
                        }
                    )
                )
            if not pts_var.empty:
                hit_parts.append(
                    pd.DataFrame(
                        {
                            "x": pts_var["mid"],
                            "y": pts_var["Value"],
                            "Date": pts_var["Date"].astype(str),
                            "Lot": pts_var["Lot"].astype(str),
                            "PredCluster": pts_var["PredCluster"],
                            "Source": "variance",
                            "Triggers": pts_var["VarianceRuleTriggers"].fillna(""),
                            "Col": col,
                            "Value": pts_var["Value"],
                            "DateTimeStr": pts_var["mid"].astype(str),
                        }
                    )
                )
            if hit_parts:
                hit_df = pd.concat(hit_parts, ignore_index=True)
                fig.add_scatter(
                    x=hit_df["x"],
                    y=hit_df["y"],
                    mode="markers",
                    name="_hitbox",
                    showlegend=False,
                    opacity=0.01,
                    marker=dict(size=30),
                    hoverinfo="skip",
                    customdata=hit_df[
                        [
                            "Date",
                            "Lot",
                            "PredCluster",
                            "Source",
                            "Triggers",
                            "Col",
                            "Value",
                            "DateTimeStr",
                        ]
                    ].values,
                )

            # 임계값 점선
            if thr_cluster is not None:
                t = thr[(thr["ClusterLabel"] == thr_cluster) & (thr["변수"] == col)]
                if not t.empty and pd.notna(t.iloc[0]["임계값"]):
                    fig.add_hline(
                        y=float(t.iloc[0]["임계값"]), line_dash="dot", line_width=2.8
                    )
            else:
                t_all = thr[
                    (thr["변수"] == col) & (thr["ClusterLabel"].isin(clusters))
                ].copy()
                t_all["임계값"] = pd.to_numeric(t_all["임계값"], errors="coerce")
                t_all = t_all.dropna(subset=["임계값"])
                if not t_all.empty:
                    idx_min = t_all["임계값"].idxmin()
                    min_val = float(t_all.loc[idx_min, "임계값"])
                    fig.add_hline(y=min_val, line_dash="dot", line_width=2.8)

            # 레이아웃
            fig.update_layout(
                margin=dict(l=0, r=0, t=32, b=0),
                xaxis_title=None,
                yaxis_title=None,
                hovermode="closest",
                clickmode="event+select",
                dragmode="select",
                spikedistance=-1,
                height=280,
            )
            fig.update_xaxes(range=[x0, x1], showspikes=False, fixedrange=True)
            if not agg.empty:
                y_min, y_max = float(agg["Value"].min()), float(agg["Value"].max())
                pad = (
                    (y_max - y_min) * 0.05
                    if y_max != y_min
                    else (abs(y_max) * 0.05 if y_max != 0 else 1.0)
                )
                fig.update_yaxes(
                    range=[y_min - pad, y_max + pad], showspikes=False, fixedrange=True
                )

            # 선택 → 즉시 해제 (nonce)
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": "reset",
                },
                key=f"plt_{col}_{st.session_state['plt_nonce']}",
                on_select="rerun",
                selection_mode=("points", "box"),
            )
            if isinstance(event, dict) and event.get("selection"):
                pts = event["selection"].get("points") or []
                picked_cd = None
                for p in pts:
                    if p.get("customdata"):
                        picked_cd = p["customdata"]
                        break
                if picked_cd is None and hit_df is not None:
                    idxs = event["selection"].get("point_indices") or []
                    if idxs:
                        i0 = int(idxs[0])
                        picked_cd = hit_df.iloc[i0][
                            [
                                "Date",
                                "Lot",
                                "PredCluster",
                                "Source",
                                "Triggers",
                                "Col",
                                "Value",
                                "DateTimeStr",
                            ]
                        ].to_list()
                if picked_cd is not None:
                    st.session_state["selected_point"] = dict(
                        Date=picked_cd[0],
                        Lot=picked_cd[1],
                        PredCluster=picked_cd[2],
                        Source=picked_cd[3],
                        Triggers=picked_cd[4],
                        Col=picked_cd[5],
                        Value=picked_cd[6],
                        DateTime=picked_cd[7],
                    )
                    st.session_state["plt_nonce"] += 1
                    st.rerun()

with right:
    # === 오른쪽 상세 패널 ===
    sel = st.session_state.get("selected_point")

    # 불량 유형/원인 매핑
    cluster_name_map = {1: "색상 불량 / 흑색 불량", 2: "도금층 불균일", 3: "수소취성"}
    cause_map = {
        1: "산처리 불충분으로 표면 산화물이 잔류해 도금 후 색상이 고르지 않게 나타남",
        2: "전류밀도 또는 전압이 최적 범위를 벗어나면 미세 균열, 핀홀, 산화물 침착 등 발생",
        3: "pH가 너무 낮거나 높을 경우 과도한 수소 이온(H⁺) 또는 수소 발생 반응이 촉진되어 금속 내부에 수소가 침투해 수소취성 발생",
    }
    # ▼ 추가: 공정최적화 텍스트 매핑
    optimize_map = {
        1: "pH 이동 표준편차 < 0.54, 저항 이동 표준편차 < 0.23, 전압 이동 표준편차  < 0.65",
        2: "저항 이동 평균 < 1.11, 전압 이동 평균의 평균 < 4.05, 전류 이동 평균의 평균 > 3.65",
        3: "pH 이동 표준편차 < 0.54, 전력 이동 표준편차 < 3.05, 전압 이동 표준편차의 평균 < 0.675",
    }

    if not sel:
        st.markdown("### 불량")
        st.info("마커(또는 박스) 선택 시 불량 상세 표시")
    else:
        # 형변환/표시 준비
        try:
            c_int = int(sel.get("PredCluster"))
        except Exception:
            c_int = None

        # 제목(크고 굵게)
        st.markdown(
            f"<div style='font-size:34px; font-weight:800; margin-bottom:6px;'>불량 {sel.get('PredCluster','')}</div>",
            unsafe_allow_html=True,
        )

        defect_name = cluster_name_map.get(c_int, "알 수 없음")
        var_disp = kor_var_name.get(sel.get("Col"), sel.get("Col", ""))
        value_disp = sel.get("Value", "")

        # LOT 구간
        sel_date = pd.to_datetime(sel.get("Date")).date() if sel.get("Date") else None
        info = lot_span[
            (lot_span["Date"] == sel_date) & (lot_span["Lot"] == sel.get("Lot"))
        ]
        lot_range = (
            f"{info.iloc[0]['start']} ~ {info.iloc[0]['end']}" if not info.empty else ""
        )

        # --- 토글 (맨 위): 불량 유형 + 원인 설명 ---
        with st.expander(f"불량 유형 : {defect_name}", expanded=False):
            cause_text = cause_map.get(c_int, "원인 정보가 없습니다.")
            st.markdown(
                f"<div style='font-size:16px; line-height:1.9; font-weight:400'>{cause_text}</div>",
                unsafe_allow_html=True,
            )

        # --- (토글 밖) 같은 (Date, Lot)의 variance 트리거 변수 모두 수집 ---
        try:
            var_rows = pred[
                (pred["Source"] == "variance")
                & (pred["Date"] == sel_date)
                & (pred["Lot"] == str(sel.get("Lot")))
            ]
            if "TriggersList" in var_rows:
                trigger_vars = sorted(
                    {
                        v
                        for L in var_rows["TriggersList"]
                        if isinstance(L, list)
                        for v in L
                    }
                )
            else:
                trigger_vars = []
        except Exception:
            trigger_vars = []

        # 본문 (라벨 20 / 값 18)
        # 날짜
        st.markdown(
            f"<div><span class='lbl'>- 날짜 : </span><span class='val'>{sel.get('Date','')}</span></div>",
            unsafe_allow_html=True,
        )
        # 발생 시간
        st.markdown(
            f"<div><span class='lbl'>- 발생 시간</span></div>", unsafe_allow_html=True
        )
        if lot_range:
            st.markdown(
                f"<div style='margin-left:16px;' class='val'>{lot_range}</div>",
                unsafe_allow_html=True,
            )
        # LOT
        st.markdown(
            f"<div><span class='lbl'>- LOT : </span><span class='val'>{sel.get('Lot','')}</span></div>",
            unsafe_allow_html=True,
        )
        # 원인 변수
        if trigger_vars:
            display_vars = [kor_var_name.get(v, v) for v in trigger_vars]
            st.markdown(
                f"<div class='lbl' style='margin-top:4px;'>- 원인 변수</div>",
                unsafe_allow_html=True,
            )
            items = "".join(
                f"<li class='val' style='line-height:1.7'>{v}</li>"
                for v in display_vars
            )
            st.markdown(
                f"<ul style='margin:6px 0 8px 22px;'>{items}</ul>",
                unsafe_allow_html=True,
            )
        # 변수명 : 값
        st.markdown(
            f"<div><span class='lbl'>- {var_disp} : </span><span class='val'>{value_disp}</span></div>",
            unsafe_allow_html=True,
        )

        # ▼ 추가: 공정최적화 (맨 아래, 기존 본문 폰트와 동일 규격)
        opt_text = optimize_map.get(c_int, "")
        if opt_text:
            st.markdown(
                f"<div><span class='lbl'>- 공정최적화 : </span>"
                f"<span class='val'>{opt_text}</span></div>",
                unsafe_allow_html=True,
            )

        # 닫기 버튼
        if st.button("닫기", key="btn_close_detail", use_container_width=True):
            st.session_state["selected_point"] = None
            st.rerun()

# ---------------------- 모델 성능(Accuracy) ----------------------
st.divider()
st.markdown("### 📊 모델 예측 성능 종합")

try:
    _gt = pd.read_csv("ground_truth_for_app.csv")
    _gt["Date"] = pd.to_datetime(_gt["Date"], errors="coerce").dt.date
    _gt["Lot"] = _gt["Lot"].astype(str)
    truth_gt = _gt.rename(columns={"ClusterLabel": "TrueCluster"})
except Exception:
    truth_gt = (
        truth_lot.copy()
        if "truth_lot" in locals()
        else pd.DataFrame(columns=["Date", "Lot", "TrueCluster"])
    )


def display_performance_pie(
    pred_df: pd.DataFrame, truth_df: pd.DataFrame, title: str, colors: dict
):
    if pred_df.empty or truth_df.empty:
        st.info(f"{title}: 예측 또는 정답 데이터 없음")
        return
    dfp = pred_df[["Date", "Lot", "PredCluster"]].copy()
    dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce").dt.date
    dfp["Lot"] = dfp["Lot"].astype(str)
    dft = truth_df[["Date", "Lot", "TrueCluster"]].copy()
    dft["Date"] = pd.to_datetime(dft["Date"], errors="coerce").dt.date
    dft["Lot"] = dft["Lot"].astype(str)

    m = pd.merge(dfp, dft, on=["Date", "Lot"], how="inner")
    if m.empty:
        st.warning(f"{title}: 평가 가능한 LOT 없음")
        return

    m["Correct"] = m["PredCluster"].astype(int) == m["TrueCluster"].astype(int)
    acc = float(m["Correct"].mean()) * 100.0
    counts = m["Correct"].value_counts().rename({True: "Correct", False: "Incorrect"})

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{title} Accuracy", value=f"{acc:.2f} %")
    with col2:
        st.metric(label="Evaluated LOTs", value=f"{len(m)} 개")

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.45,
        color=counts.index,
        color_discrete_map=colors,
        title=f"{title} 모델 성능",
    )
    fig.update_traces(
        textposition="outside",
        textinfo="percent+label",
        textfont_size=18,
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig.update_layout(
        showlegend=False,
        title_font_size=20,
        margin=dict(t=70, b=120, l=70, r=70),
        height=420,
        uniformtext_minsize=16,
        uniformtext_mode="show",
    )
    fig.add_annotation(
        text=f"{acc:.1f}%", x=0.5, y=0.5, showarrow=False, font=dict(size=22)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


c3, c4 = st.columns(2)
with c3:
    display_performance_pie(
        pred[pred["Source"] == "lgbm"].copy(),
        truth_gt.copy(),
        "LGBM",
        {"Correct": "#1f77b4", "Incorrect": "#ff7f0e"},
    )
with c4:
    display_performance_pie(
        pred[pred["Source"] == "variance"].copy(),
        truth_gt.copy(),
        "Variance",
        {"Correct": "#2ca02c", "Incorrect": "#d62728"},
    )

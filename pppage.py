import streamlit as st


def show(selected_date=None, date_range=None):
    # overlay_app.py
    import pandas as pd
    import plotly.express as px

    # 대시보드 제목
    st.title("불량 분석 대시보드")

    st.markdown(
        """<style>
    /* 레이아웃 여백/헤더 숨김 */
    [data-testid="stAppViewContainer"] .main .block-container {padding:0rem;}
    [data-testid="stHeader"]{display:none;} footer{visibility:hidden;}

    /* 기본 라벨/값 규칙 */
    .lbl {font-size:20px; font-weight:700;}
    .val {font-size:18px; font-weight:400;}

    /* Expander(토글) 스타일 */
    div[data-testid="stExpander"] details summary {
    font-size: 26px !important;
    font-weight: 1000 !important;
    line-height: 1.3 !important;
    }
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
        df = pd.read_csv("df_sensor.csv")
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        return df.dropna(subset=["DateTime"]).sort_values("DateTime")

    @st.cache_data
    def load_thresholds():
        df = pd.read_csv("df_thresh_z (1).csv")
        df["ClusterLabel"] = pd.to_numeric(df["ClusterLabel"], errors="coerce").astype(
            "Int64"
        )
        df["임계값"] = pd.to_numeric(df["임계값"], errors="coerce")
        if "direction" not in df.columns:
            df["direction"] = 1
        return df

    @st.cache_data
    def load_preds():
        frames = []
        # LGBM: pred_lot.csv (PredCluster 0/1/2/3)
        try:
            a = pd.read_csv("pred_lot.csv")
            a["Source"] = "lgbm"
            a["PredCluster"] = (
                pd.to_numeric(a["PredCluster"], errors="coerce").fillna(0).astype(int)
            )
            a["PredFlag"] = None
            frames.append(a[["Date", "Lot", "PredCluster", "PredFlag", "Source"]])
        except FileNotFoundError:
            pass

        # 이상치 모델: pred_binary_lot.csv (PredFlag 0/1)
        try:
            b = pd.read_csv("pred_binary_lot.csv")
            b["Source"] = "anomaly"
            b["PredFlag"] = (
                pd.to_numeric(b["PredFlag"], errors="coerce").fillna(0).astype(int)
            )
            if "PredCluster" not in b.columns:
                b["PredCluster"] = None
            frames.append(b[["Date", "Lot", "PredCluster", "PredFlag", "Source"]])
        except FileNotFoundError:
            pass

        if not frames:
            return pd.DataFrame(
                columns=["Date", "Lot", "PredCluster", "PredFlag", "Source"]
            )

        pred_all = pd.concat(frames, ignore_index=True)
        pred_all["Date"] = pd.to_datetime(pred_all["Date"], errors="coerce").dt.date
        pred_all["Lot"] = pred_all["Lot"].astype(str)
        return pred_all

    raw = load_raw()
    thr = load_thresholds()
    pred = load_preds()

    # 키 정규화
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce").dt.date
    raw["Lot"] = raw["Lot"].astype(str)

    # -------------------- 성능용 정답 라벨 로더 --------------------
    @st.cache_data
    def load_truth_lot(raw_df: pd.DataFrame) -> pd.DataFrame:
        if "ClusterLabel" in raw_df.columns:
            tru = raw_df.dropna(subset=["ClusterLabel"]).copy()
            tru["ClusterLabel"] = pd.to_numeric(tru["ClusterLabel"], errors="coerce")
            tru = tru.dropna(subset=["ClusterLabel"])
            if not tru.empty:
                g = tru.groupby(["Date", "Lot"])["ClusterLabel"].max().reset_index()
                g = g.rename(columns={"ClusterLabel": "TrueCluster"})
                g["Lot"] = g["Lot"].astype(str)
                g["Date"] = pd.to_datetime(g["Date"], errors="coerce").dt.date
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

    # LOT 시간 구간
    lot_span = (
        raw.groupby(["Date", "Lot"])["DateTime"]
        .agg(start="min", end="max")
        .reset_index()
    )

    # --------------------------- UI 상단 ---------------------------
    c1, c2 = st.columns([1, 1])
    with c1:
        mode = st.selectbox("기간 모드", ["전체", "기간선택"], index=0)
        limit_start, limit_end = (
            raw["DateTime"].min().date(),
            raw["DateTime"].max().date(),
        )

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

    # ---------------------- LGBM + 이상치 LOT 상태 병합 ----------------------
    pred_lgbm = pred[pred["Source"] == "lgbm"][
        ["Date", "Lot", "PredCluster"]
    ].drop_duplicates()
    pred_anom = pred[pred["Source"] == "anomaly"][
        ["Date", "Lot", "PredFlag"]
    ].drop_duplicates()

    lot_status = lot_span.merge(pred_lgbm, on=["Date", "Lot"], how="left").merge(
        pred_anom, on=["Date", "Lot"], how="left"
    )
    lot_status["PredCluster"] = (
        pd.to_numeric(lot_status["PredCluster"], errors="coerce").fillna(0).astype(int)
    )
    lot_status["PredFlag"] = (
        pd.to_numeric(lot_status["PredFlag"], errors="coerce").fillna(0).astype(int)
    )

    # ------------------------ 상태 초기화 ------------------------
    if "selected_point" not in st.session_state:
        st.session_state["selected_point"] = None
    if "plt_nonce" not in st.session_state:
        st.session_state["plt_nonce"] = 0  # 선택 즉시 해제용

    # ---------------- 변수 한글명 / 표시 변수 ----------------
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

    # -------------------- PredCluster 기준 임계값 함수 --------------------
    def get_threshold_for(cluster_label: int, var: str, thr_df: pd.DataFrame):
        """
        특정 클러스터와 변수에 대한 임계값과 방향(direction)을 반환.
        - direction 없으면 +1 가정
        - 동일 (cluster,var)에 여러 행이 있으면 effective=(임계값*direction)이 가장 작은 행 선택
        """
        t = thr_df[
            (thr_df["ClusterLabel"] == cluster_label) & (thr_df["변수"] == var)
        ].copy()
        if t.empty:
            return None, 1
        if "direction" not in t.columns:
            t["direction"] = 1
        t["direction"] = (
            pd.to_numeric(t["direction"], errors="coerce").fillna(1).astype(int)
        )
        t["임계값"] = pd.to_numeric(t["임계값"], errors="coerce")
        t["effective"] = t["임계값"] * t["direction"]
        t = t.dropna(subset=["임계값", "effective"])
        if t.empty:
            return None, 1
        best = t.loc[t["effective"].idxmin()]
        return float(best["임계값"]), int(best["direction"])

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
            # LOT 시간 중앙값(mid)
            span_mid = (
                base.groupby(["Date", "Lot"], as_index=False)["DateTime"]
                .agg(start="min", end="max")
                .assign(mid=lambda d: d["start"] + (d["end"] - d["start"]) / 2)
            )
            span_mid = span_mid.merge(
                base[["Date", "Lot", "LotNum"]].drop_duplicates(),
                on=["Date", "Lot"],
                how="left",
            )

            # LOT별 상태(PredCluster, PredFlag) 붙이기
            span_mid = span_mid.merge(
                lot_status[["Date", "Lot", "PredCluster", "PredFlag"]],
                on=["Date", "Lot"],
                how="left",
            )

            for col in ycols:
                # LOT 평균값 계산
                agg_val = (
                    base.groupby(["Date", "Lot"], as_index=False)[col]
                    .mean()
                    .rename(columns={col: "Value"})
                )
                agg = agg_val.merge(
                    span_mid, on=["Date", "Lot"], how="left"
                ).sort_values("mid")

                fig = px.line(agg, x="mid", y="Value", title=kor_var_name.get(col, col))

                # 1) 하이라이트 밴드: 이상치(=1) + 선택유형 LOT만 → 반투명 빨간 사각형
                bands = agg[
                    (agg["PredFlag"] == 1) & (agg["PredCluster"].isin(clusters))
                ][["Date", "Lot", "start", "end"]].drop_duplicates()
                for _, r in bands.iterrows():
                    fig.add_vrect(
                        x0=r["start"],
                        x1=r["end"],
                        opacity=0.15,
                        line_width=0,
                        fillcolor="rgba(255,0,0,0.10)",
                    )

                # 2) LGBM 유형 포인트(삼각형) — 사이즈 ↑ (9 → 12)
                pts_lgbm = agg[agg["PredCluster"].isin(clusters)].dropna(
                    subset=["Value"]
                )
                if not pts_lgbm.empty:
                    fig.add_scatter(
                        x=pts_lgbm["mid"],
                        y=pts_lgbm["Value"],
                        mode="markers",
                        name="LGBM 유형",
                        marker=dict(size=12, symbol="triangle-up"),
                        customdata=pts_lgbm[
                            ["Date", "Lot", "PredCluster", "PredFlag", "Value"]
                        ].values,
                        hovertemplate="시간=%{x}<br>값=%{y}<br>LGBM 유형=%{customdata[2]}<br>이상치(PredFlag)=%{customdata[3]}<extra></extra>",
                    )

                # 3) 이상치모델 + LGBM 점: (PredFlag==1) 이면서 해당 LOT의 LGBM PredCluster 임계/방향 초과
                marks = []
                for _, row in agg.iterrows():
                    c = int(row.get("PredCluster", 0))
                    f = int(row.get("PredFlag", 0))
                    if f != 1 or c == 0 or pd.isna(row["Value"]):
                        continue
                    thr_val, direction = get_threshold_for(c, col, thr)
                    if thr_val is None:
                        continue
                    if direction * (row["Value"] - thr_val) > 0:
                        marks.append(row)
                if marks:
                    mdf = pd.DataFrame(marks)
                    fig.add_scatter(
                        x=mdf["mid"],
                        y=mdf["Value"],
                        mode="markers",
                        name="이상치모델 + LGBM",
                        marker=dict(size=10, symbol="x", color="red"),
                        customdata=mdf[
                            ["Date", "Lot", "PredCluster", "PredFlag", "Value"]
                        ].values,
                        hovertemplate="시간=%{x}<br>값=%{y}<br>LGBM 유형=%{customdata[2]}<br>이상치(PredFlag)=%{customdata[3]}<extra></extra>",
                    )

                # 4) 임계값 점선
                if thr_cluster is not None:
                    tv, _dir = get_threshold_for(thr_cluster, col, thr)
                    if tv is not None:
                        fig.add_hline(y=tv, line_dash="dot", line_width=2.8)
                else:
                    # 다중 선택일 땐 해당 클러스터들의 임계 중 가장 엄격한(효과값 최소) 하나만 표시
                    t_all = thr[
                        (thr["변수"] == col) & (thr["ClusterLabel"].isin(clusters))
                    ].copy()
                    if not t_all.empty:
                        t_all["effective"] = pd.to_numeric(
                            t_all["임계값"], errors="coerce"
                        ) * pd.to_numeric(
                            t_all.get("direction", 1), errors="coerce"
                        ).fillna(
                            1
                        )
                        t_all = t_all.dropna(subset=["effective"])
                        if not t_all.empty:
                            best = t_all.loc[t_all["effective"].idxmin()]
                            fig.add_hline(
                                y=float(best["임계값"]), line_dash="dot", line_width=2.8
                            )

                # 레이아웃/축
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
                        range=[y_min - pad, y_max + pad],
                        showspikes=False,
                        fixedrange=True,
                    )

                # 선택 이벤트
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
                        if p.get("customdata") is not None:
                            picked_cd = p["customdata"]
                            break
                    if picked_cd is not None:
                        st.session_state["selected_point"] = dict(
                            Date=str(picked_cd[0]),
                            Lot=str(picked_cd[1]),
                            PredCluster=(
                                int(picked_cd[2]) if picked_cd[2] is not None else 0
                            ),
                            PredFlag=(
                                int(picked_cd[3]) if picked_cd[3] is not None else 0
                            ),
                            Col=col,
                            Value=(
                                float(picked_cd[4])
                                if picked_cd[4] is not None
                                else None
                            ),
                        )
                        st.session_state["plt_nonce"] += 1
                        st.rerun()

    with right:
        # === 오른쪽 상세 패널 ===
        sel = st.session_state.get("selected_point")

        # 불량 유형/원인 매핑
        cluster_name_map = {
            1: "색상 불량 / 흑색 불량",
            2: "도금층 불균일",
            3: "수소취성",
        }
        cause_map = {
            1: "산처리 불충분으로 표면 산화물이 잔류해 도금 후 색상이 고르지 않게 나타남",
            2: "전류밀도 또는 전압이 최적 범위를 벗어나면 미세 균열, 핀홀, 산화물 침착 등 발생",
            3: "pH가 너무 낮거나 높을 경우 과도한 수소 이온(H⁺) 또는 수소 발생 반응이 촉진되어 금속 내부에 수소가 침투해 수소취성 발생",
        }
        optimize_map = {
            1: "pH 이동 표준편차 < 0.54, 저항 이동 표준편차 < 0.23, 전압 이동 표준편차  < 0.65",
            2: "저항 이동 평균 < 1.11, 전압 이동 평균의 평균 < 4.05, 전류 이동 평균의 평균 > 3.65",
            3: "pH 이동 표준편차 < 0.54, 전력 이동 표준편차 < 3.05, 전압 이동 표준편차의 평균 < 0.675",
        }

        if not sel:
            st.markdown("### 불량")
            st.info("마커(또는 박스) 선택 시 불량 상세 표시")
        else:
            try:
                c_int = int(sel.get("PredCluster"))
            except Exception:
                c_int = 0

            st.markdown(
                f"<div style='font-size:34px; font-weight:800; margin-bottom:6px;'>불량 {c_int}</div>",
                unsafe_allow_html=True,
            )

            defect_name = cluster_name_map.get(c_int, "알 수 없음")
            var_disp = kor_var_name.get(sel.get("Col"), sel.get("Col", ""))
            value_disp = sel.get("Value", "")

            # 이 LOT의 시간 범위
            sel_date = (
                pd.to_datetime(sel.get("Date")).date() if sel.get("Date") else None
            )
            sel_lot = str(sel.get("Lot"))
            info = lot_span[
                (lot_span["Date"] == sel_date) & (lot_span["Lot"] == sel_lot)
            ]
            lot_range = (
                f"{info.iloc[0]['start']} ~ {info.iloc[0]['end']}"
                if not info.empty
                else ""
            )

            with st.expander(f"불량 유형 : {defect_name}", expanded=False):
                cause_text = cause_map.get(c_int, "원인 정보가 없습니다.")
                st.markdown(
                    f"<div style='font-size:16px; line-height:1.9; font-weight:400'>{cause_text}</div>",
                    unsafe_allow_html=True,
                )

            # --- 원인 변수: 실시간 계산 (이 LOT의 변수 평균 vs PredCluster 임계) ---
            cause_vars = []
            if c_int != 0:
                lot_df = raw[(raw["Date"] == sel_date) & (raw["Lot"] == sel_lot)]
                for v in ycols:
                    if v not in lot_df.columns:
                        continue
                    mean_v = lot_df[v].mean()
                    thr_val, d = get_threshold_for(c_int, v, thr)
                    if thr_val is None:
                        continue
                    if d * (mean_v - thr_val) > 0:
                        cause_vars.append(kor_var_name.get(v, v))

            # 본문
            st.markdown(
                f"<div><span class='lbl'>- 날짜 : </span><span class='val'>{sel.get('Date','')}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div><span class='lbl'>- 발생 시간</span></div>",
                unsafe_allow_html=True,
            )
            if lot_range:
                st.markdown(
                    f"<div style='margin-left:16px;' class='val'>{lot_range}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div><span class='lbl'>- LOT : </span><span class='val'>{sel.get('Lot','')}</span></div>",
                unsafe_allow_html=True,
            )

            if cause_vars:
                st.markdown(
                    f"<div class='lbl' style='margin-top:4px;'>- 원인 변수</div>",
                    unsafe_allow_html=True,
                )
                items = "".join(
                    f"<li class='val' style='line-height:1.7'>{v}</li>"
                    for v in cause_vars
                )
                st.markdown(
                    f"<ul style='margin:6px 0 8px 22px;'>{items}</ul>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div><span class='lbl'>- {var_disp} : </span><span class='val'>{value_disp}</span></div>",
                unsafe_allow_html=True,
            )

            opt_text = optimize_map.get(c_int, "")
            if opt_text:
                st.markdown(
                    f"<div><span class='lbl'>- 공정최적화 : </span><span class='val'>{opt_text}</span></div>",
                    unsafe_allow_html=True,
                )

            if st.button("닫기", key="btn_close_detail", use_container_width=True):
                st.session_state["selected_point"] = None
                st.rerun()

    # ---------------------- 성능(Accuracy) ----------------------
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
        counts = (
            m["Correct"].value_counts().rename({True: "Correct", False: "Incorrect"})
        )
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

    def display_performance_pie_binary(
        pred_flag_df: pd.DataFrame, truth_df: pd.DataFrame, title: str, colors: dict
    ):
        if pred_flag_df.empty or truth_df.empty:
            st.info(f"{title}: 예측 또는 정답 데이터 없음")
            return
        p = pred_flag_df[["Date", "Lot", "PredFlag"]].copy()
        p["Date"] = pd.to_datetime(p["Date"], errors="coerce").dt.date
        p["Lot"] = p["Lot"].astype(str)

        t = truth_df[["Date", "Lot", "TrueCluster"]].copy()
        t["Date"] = pd.to_datetime(t["Date"], errors="coerce").dt.date
        t["Lot"] = t["Lot"].astype(str)
        t["TrueBin"] = (
            pd.to_numeric(t["TrueCluster"], errors="coerce").fillna(0).astype(int) > 0
        ).astype(int)

        m = pd.merge(p, t[["Date", "Lot", "TrueBin"]], on=["Date", "Lot"], how="inner")
        if m.empty:
            st.warning(f"{title}: 평가 가능한 LOT 없음")
            return

        m["Correct"] = m["PredFlag"].astype(int) == m["TrueBin"].astype(int)
        acc = float(m["Correct"].mean()) * 100.0
        counts = (
            m["Correct"].value_counts().rename({True: "Correct", False: "Incorrect"})
        )

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
        display_performance_pie_binary(
            pred[pred["Source"] == "anomaly"].copy(),
            truth_gt.copy(),
            "이상치",  # ← ‘Anomaly’ → ‘이상치’ (결과: “이상치 모델 성능”)
            {"Correct": "#2ca02c", "Incorrect": "#d62728"},
        )


if __name__ == "__main__":
    pass

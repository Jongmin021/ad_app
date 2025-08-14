import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import streamlit as st
import pandas as pd  # 필요 시


def show(selected_date=None, date_range=None):
    # 'df_final.pkl' 파일을 읽어 데이터프레임으로 저장
    try:
        df_final = pd.read_pickle(r"df_final.pkl")
    except FileNotFoundError:
        st.error("파일을 찾을 수 없습니다. 'df_final.pkl' 파일 경로를 확인해주세요.")
        st.stop()

    # --- CSS (추가/교체): 제목/소제목 왼쪽 정렬 ---
    st.markdown(
        """
    <style>
    /* 제목 크기 통일: 대/중 */
    .title-xl { font-size: 34px; font-weight: 800; margin: 6px 0 14px; text-align: left; }
    .title-md { font-size: 22px; font-weight: 700; margin: 18px 0 10px; text-align: left; }
    .section-title { font-size: 28px; font-weight: 700; text-align: left; margin-bottom: 10px;}

    /* KPI 카드 (정사각형) */
    .kpi-box {
        background: #fff;
        border: 1px solid #eef0f2;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.1);
        aspect-ratio: 1/1;
        padding: 10px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    .kpi-value { font-size: 20px; font-weight: 800; line-height: 1.2; }
    .kpi-label { font-size: 15px; color: gray; }

    /* 불량 유형별 Lot 수 표시를 위한 CSS */
    .defect-type-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-top: 10px;
        border: 1px solid #eef0f2;
        border-radius: 10px;
    }
    .defect-type-item {
        text-align: center;
        padding: 15px 0;
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .defect-type-label {
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .defect-type-value {
        font-size: 24px;
        font-weight: 800;
    }
    .defect-type-sub {
        font-size: 14px;
        color: gray;
    }

    /* 세로선 */
    .vertical-separator {
        border-right: 1px solid #eef0f2;
    }
    .vertical-separator.left {
        border-left: 1px solid #eef0f2;
    }

    /* 그래프 컨테이너 */
    .chart-container {
        border: 1px solid #eef0f2;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.1);
        padding: 10px;
        background: #fff;
        height: 100%;
    }

    /* Streamlit 슬라이더 트랙 색상 변경 */
    div[data-testid="stSlider"] > div[role="slider"] > div[data-testid="stSliderTrack"] > div {
        background-color: black;
    }
    div[data-testid="stSlider"] > div[role="slider"] > div[data-testid="stSliderThumb"] {
        background-color: black;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────
    # 유틸 함수
    # ─────────────────────────────────────────────────────────
    def compute_mtbd_days(df: pd.DataFrame) -> float | None:
        """MTBD (Mean Time Between Defects) - 날짜 단위"""
        s = pd.to_datetime(df.loc[df["IsDefect"] > 0, "Date"]).sort_values()
        diffs = s.diff().dropna()
        if diffs.empty:
            return None
        return diffs.dt.total_seconds().mean() / 86400.0

    def fmt_num(x, pattern="{:.1f}", default="—"):
        try:
            return pattern.format(float(x))
        except (TypeError, ValueError):
            return default

    def render_kpi_card(title, value, color="black"):
        """KPI 카드"""
        st.markdown(
            f"""
            <div style="
                padding: 15px; 
                border-radius: 10px; 
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                text-align: center;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100%;
                aspect-ratio: 1/1;
                background-color: white;
                color: black;
            ">
                <h5 style="color: gray; margin: 0 0 10px 0;">{title}</h5>
                <div style="font-size: 2.5em; font-weight: bold; color: {color};">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────
    # 집계/지표 계산
    # ─────────────────────────────────────────────────────────
    base = (
        df_final.drop_duplicates(["Date", "Lot"])
        if {"Date", "Lot"}.issubset(df_final.columns)
        else df_final.copy()
    )
    total_lot = int(len(base))
    defect_lot = int((base["IsDefect"] > 0).sum())
    defect_rate = (defect_lot / total_lot) if total_lot else 0.0
    mtbd = compute_mtbd_days(base)

    type_counts = (
        base.loc[base["IsDefect"] > 0, "ClusterLabel"]
        .replace(0, pd.NA)
        .dropna()
        .astype(int)
        .value_counts()
        .reindex([1, 2, 3], fill_value=0)
        .to_dict()
    )
    total_defect_lot = defect_lot

    # 날짜 범위
    start_date = pd.to_datetime(base["Date"].min()).date()
    end_date = pd.to_datetime(base["Date"].max()).date()

    # ─────────────────────────────────────────────────────────
    # 상단 타이틀(왼쪽 정렬)
    # ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align: left; margin-bottom: 20px;">
            <h1 style="font-size: 40px; font-weight: 800; color: #333; margin:0;">산제 전처리 공정 대시보드</h1>
            <div style="display: flex; align-items: center; font-size: 16px; margin-top: 6px; color:#999;">
                <div style="font-weight: bold; margin-right: 10px;">실시간 공정 모니터링 시스템</div>
                <div style="color: #bbb; font-size: 20px; margin: 0 10px;">|</div>
                <div><span>{start_date} ~ {end_date}</span></div>
            </div>
        </div>
    """.format(
            start_date=start_date, end_date=end_date
        ),
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # 품질 KPI
    # ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">품질 KPI</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    cols = st.columns(4)

    with cols[0]:
        st.markdown('<div style="margin-top: 20px;">', unsafe_allow_html=True)
        counts = base["IsDefect"].value_counts().reindex([0, 1], fill_value=0)
        labels = ["정상", "불량"]
        colors = ["green", "red"]

        fig = go.Figure(data=[go.Pie(labels=labels, values=counts, hole=0.7)])
        fig.update_traces(
            marker=dict(colors=colors),
            hoverinfo="label+value+percent",
            textinfo="none",
        )
        fig.update_layout(
            showlegend=True,
            margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
            height=320,
            width=320,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        render_kpi_card("총 LOT", f"{total_lot:,}", color="black")

    with cols[2]:
        render_kpi_card("불량 LOT", f"{defect_lot:,}", color="red")

    with cols[3]:
        render_kpi_card("MTBD", f"{fmt_num(mtbd)}일", color="black")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # 센서 데이터 추이 (df_final, pH/저항/전력)
    # ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">센서 데이터 추이</div>', unsafe_allow_html=True
    )

    # 6기간 이동평균/표준편차 계산(없으면 생성)
    if "pH_ma6" not in df_final.columns:
        df_final["pH_ma6"] = df_final["pH"].rolling(window=6, min_periods=1).mean()
        df_final["pH_std6"] = df_final["pH"].rolling(window=6, min_periods=1).std()
        df_final["Resistance_ma6"] = (
            df_final["Resistance"].rolling(window=6, min_periods=1).mean()
        )
        df_final["Resistance_std6"] = (
            df_final["Resistance"].rolling(window=6, min_periods=1).std()
        )
        df_final["Power_ma6"] = (
            df_final["Power"].rolling(window=6, min_periods=1).mean()
        )
        df_final["Power_std6"] = (
            df_final["Power"].rolling(window=6, min_periods=1).std()
        )

    # 조회 날짜 선택 → 해당 날짜 데이터만
    min_date = pd.to_datetime(df_final["Date"]).min().date()
    max_date = pd.to_datetime(df_final["Date"]).max().date()
    selected_date = st.date_input(
        "",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="date_selector_main",
    )
    selected_df = df_final[
        pd.to_datetime(df_final["Date"]).dt.date == selected_date
    ].copy()

    if not selected_df.empty:
        selected_df = selected_df.sort_values(by="Lot")

        # 3개 그래프: pH / 저항 / 전력
        cols_sensor = st.columns(3)

        # 색상 팔레트(그래프별 다른 계열)
        PH_COLORS = ("#3B82F6", "#1D4ED8", "#1E3A8A")  # 원시/MA/STD
        RES_COLORS = ("#EF4444", "#B91C1C", "#7F1D1D")
        POW_COLORS = ("#22C55E", "#15803D", "#14532D")

        with cols_sensor[0]:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["pH"],
                    mode="lines+markers",
                    name="pH",
                    marker_symbol="circle",
                    line=dict(color=PH_COLORS[0]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["pH_ma6"],
                    mode="lines+markers",
                    name="pH 이동 평균",
                    marker_symbol="square",
                    line=dict(color=PH_COLORS[1]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["pH_std6"],
                    mode="lines+markers",
                    name="pH 이동 표준편차",
                    marker_symbol="diamond",
                    line=dict(color=PH_COLORS[2], dash="dot"),
                )
            )
            fig.update_layout(
                title_text="<b>pH</b>",
                title_font_size=20,
                title_x=0.5,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5
                ),
                xaxis_title="Lot",
            )
            st.plotly_chart(fig, use_container_width=True)

        with cols_sensor[1]:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["Resistance"],
                    mode="lines+markers",
                    name="저항",
                    marker_symbol="circle",
                    line=dict(color=RES_COLORS[0]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["Resistance_ma6"],
                    mode="lines+markers",
                    name="저항 이동 평균",
                    marker_symbol="square",
                    line=dict(color=RES_COLORS[1]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["Resistance_std6"],
                    mode="lines+markers",
                    name="저항 이동 표준편차",
                    marker_symbol="diamond",
                    line=dict(color=RES_COLORS[2], dash="dot"),
                )
            )
            fig.update_layout(
                title_text="<b>저항</b>",
                title_font_size=20,
                title_x=0.5,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5
                ),
                xaxis_title="Lot",
            )
            st.plotly_chart(fig, use_container_width=True)

        with cols_sensor[2]:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["Power"],
                    mode="lines+markers",
                    name="전력",
                    marker_symbol="circle",
                    line=dict(color=POW_COLORS[0]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["Power_ma6"],
                    mode="lines+markers",
                    name="전력 이동 평균",
                    marker_symbol="square",
                    line=dict(color=POW_COLORS[1]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_df["Lot"],
                    y=selected_df["Power_std6"],
                    mode="lines+markers",
                    name="전력 이동 표준편차",
                    marker_symbol="diamond",
                    line=dict(color=POW_COLORS[2], dash="dot"),
                )
            )
            fig.update_layout(
                title_text="<b>전력</b>",
                title_font_size=20,
                title_x=0.5,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5
                ),
                xaxis_title="Lot",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # df_sensor.csv (온도/전압/전류) — 선택 날짜만 필터
    # 롤링(window=6) 먼저 계산 → 그 다음 Lot별 평균으로 그래프 표시 (pH 형식 그대로)
    # ─────────────────────────────────────────────────────────
    SENSOR_CSV = "/Users/t2023-m0056/Desktop/파일/df_sensor.csv"

    try:
        df_sensor = pd.read_csv(SENSOR_CSV)
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {SENSOR_CSV}")
        df_sensor = None

    if df_sensor is not None:
        required_cols = {"DateTime", "Lot", "Temp", "Voltage", "Current"}
        missing = required_cols - set(df_sensor.columns)
        if missing:
            st.error(f"df_sensor.csv 필수 컬럼 누락: {missing}")
        else:
            # 시간 파싱 및 정렬
            df_sensor["DateTime"] = pd.to_datetime(
                df_sensor["DateTime"], errors="coerce"
            )
            df_sensor = df_sensor.dropna(subset=["DateTime"]).sort_values(
                ["DateTime", "Lot"]
            )

            # ▶ 달력에서 고른 selected_date만 표시
            day_df = df_sensor[df_sensor["DateTime"].dt.date == selected_date].copy()

            if day_df.empty:
                st.error("선택한 날짜에 해당하는 데이터가 없습니다.")
            else:
                # 시간 순서대로 롤링 계산(먼저)
                day_df = day_df.sort_values(["DateTime", "Lot"]).reset_index(drop=True)
                for c in ["Temp", "Voltage", "Current"]:
                    day_df[f"{c}_ma6"] = (
                        day_df[c].rolling(window=6, min_periods=6).mean()
                    )
                    day_df[f"{c}_std6"] = (
                        day_df[c].rolling(window=6, min_periods=6).std()
                    )

                # 그 다음 Lot별 평균값으로 집계 (원시/MA/STD 각각)
                def lot_avg(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
                    return df_in.groupby("Lot", as_index=False)[col].mean()

                # ─ 온도: 원시/MA/STD Lot평균 테이블을 병합
                temp_raw = lot_avg(day_df, "Temp")
                temp_ma = lot_avg(day_df, "Temp_ma6")
                temp_st = lot_avg(day_df, "Temp_std6")
                temp_agg = (
                    temp_raw.merge(temp_ma, on="Lot", how="left")
                    .merge(temp_st, on="Lot", how="left")
                    .sort_values("Lot")
                )

                # ─ 전압
                volt_raw = lot_avg(day_df, "Voltage")
                volt_ma = lot_avg(day_df, "Voltage_ma6")
                volt_st = lot_avg(day_df, "Voltage_std6")
                volt_agg = (
                    volt_raw.merge(volt_ma, on="Lot", how="left")
                    .merge(volt_st, on="Lot", how="left")
                    .sort_values("Lot")
                )

                # ─ 전류
                curr_raw = lot_avg(day_df, "Current")
                curr_ma = lot_avg(day_df, "Current_ma6")
                curr_st = lot_avg(day_df, "Current_std6")
                curr_agg = (
                    curr_raw.merge(curr_ma, on="Lot", how="left")
                    .merge(curr_st, on="Lot", how="left")
                    .sort_values("Lot")
                )

                st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
                cols_sensor2 = st.columns(3)

                # 색상 팔레트(그래프별 다른 계열)
                TEMP_COLORS = ("#F59E0B", "#B45309", "#7C2D12")  # 원시/MA/STD
                VOLT_COLORS = ("#8B5CF6", "#6D28D9", "#4C1D95")
                CURR_COLORS = ("#E3D80A", "#E2D70D", "#FBDF0C")

                # ─ 온도 ─ (pH 형식 그대로: 원시/MA/STD)
                with cols_sensor2[0]:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=temp_agg["Lot"],
                            y=temp_agg["Temp"],
                            mode="lines+markers",
                            name="온도",
                            marker_symbol="circle",
                            line=dict(color=TEMP_COLORS[0]),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=temp_agg["Lot"],
                            y=temp_agg["Temp_ma6"],
                            mode="lines+markers",
                            name="온도 이동 평균",
                            marker_symbol="square",
                            line=dict(color=TEMP_COLORS[1]),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=temp_agg["Lot"],
                            y=temp_agg["Temp_std6"],
                            mode="lines+markers",
                            name="온도 이동 표준편차",
                            marker_symbol="diamond",
                            line=dict(color=TEMP_COLORS[2], dash="dot"),
                        )
                    )
                    fig.update_layout(
                        title_text="<b>온도</b>",
                        title_font_size=20,
                        title_x=0.5,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.5,
                            xanchor="center",
                            x=0.5,
                        ),
                        xaxis_title="Lot",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ─ 전압 ─
                with cols_sensor2[1]:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=volt_agg["Lot"],
                            y=volt_agg["Voltage"],
                            mode="lines+markers",
                            name="전압",
                            marker_symbol="circle",
                            line=dict(color=VOLT_COLORS[0]),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=volt_agg["Lot"],
                            y=volt_agg["Voltage_ma6"],
                            mode="lines+markers",
                            name="전압 이동 평균",
                            marker_symbol="square",
                            line=dict(color=VOLT_COLORS[1]),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=volt_agg["Lot"],
                            y=volt_agg["Voltage_std6"],
                            mode="lines+markers",
                            name="전압 이동 표준편차",
                            marker_symbol="diamond",
                            line=dict(color=VOLT_COLORS[2], dash="dot"),
                        )
                    )
                    fig.update_layout(
                        title_text="<b>전압</b>",
                        title_font_size=20,
                        title_x=0.5,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.5,
                            xanchor="center",
                            x=0.5,
                        ),
                        xaxis_title="Lot",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ─ 전류 ─
                with cols_sensor2[2]:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=curr_agg["Lot"],
                            y=curr_agg["Current"],
                            mode="lines+markers",
                            name="전류",
                            marker_symbol="circle",
                            line=dict(color=CURR_COLORS[0]),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=curr_agg["Lot"],
                            y=curr_agg["Current_ma6"],
                            mode="lines+markers",
                            name="전류 이동 평균",
                            marker_symbol="square",
                            line=dict(color=CURR_COLORS[1]),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=curr_agg["Lot"],
                            y=curr_agg["Current_std6"],
                            mode="lines+markers",
                            name="전류 이동 표준편차",
                            marker_symbol="diamond",
                            line=dict(color=CURR_COLORS[2], dash="dot"),
                        )
                    )
                    fig.update_layout(
                        title_text="<b>전류</b>",
                        title_font_size=20,
                        title_x=0.5,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.5,
                            xanchor="center",
                            x=0.5,
                        ),
                        xaxis_title="Lot",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    # 불량 LOT 상세
    # ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">불량 LOT 상세 </div>', unsafe_allow_html=True
    )
    st.markdown('<div style="height:5px;"></div>', unsafe_allow_html=True)

    lot_count_1 = type_counts.get(1, 0)
    lot_count_2 = type_counts.get(2, 0)
    lot_count_3 = type_counts.get(3, 0)

    defect_percentage_1 = (
        (lot_count_1 / total_defect_lot * 100) if total_defect_lot > 0 else 0
    )
    defect_percentage_2 = (
        (lot_count_2 / total_defect_lot * 100) if total_defect_lot > 0 else 0
    )
    defect_percentage_3 = (
        (lot_count_3 / total_defect_lot * 100) if total_defect_lot > 0 else 0
    )

    st.markdown(
        f"""
    <div class="defect-type-container">
        <div class="vertical-separator left"></div>
        <div class="defect-type-item">
            <div class="defect-type-value">{lot_count_1} LOT</div>
            <div class="defect-type-sub">불량 유형 1 ({defect_percentage_1:.1f}%)</div>
        </div>
        <div class="vertical-separator"></div>
        <div class="defect-type-item">
            <div class="defect-type-value">{lot_count_2} LOT</div>
            <div class="defect-type-sub">불량 유형 2 ({defect_percentage_2:.1f}%)</div>
        </div>
        <div class="vertical-separator"></div>
        <div class="defect-type-item">
            <div class="defect-type-value">{lot_count_3} LOT</div>
            <div class="defect-type-sub">불량 유형 3 ({defect_percentage_3:.1f}%)</div>
        </div>
        <div class="vertical-separator"></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────
    # 불량 발생 추이 (평일만)
    # ─────────────────────────────────────────────────────────
    df_final["Date_norm"] = pd.to_datetime(df_final["Date"]).dt.normalize()
    df_final["dayofweek"] = df_final["Date_norm"].dt.dayofweek

    start_day = df_final["Date_norm"].min()
    end_day = df_final["Date_norm"].max()
    full_weekdays = pd.date_range(start_day, end_day, freq="B")

    weekly_defects = (
        df_final[
            (df_final["Date_norm"] >= start_day)
            & (df_final["Date_norm"] <= end_day)
            & (df_final["dayofweek"] < 5)
        ]
        .loc[lambda df: df["IsDefect"] == 1]
        .groupby("Date_norm")
        .size()
    )

    weekly_defects = weekly_defects.reindex(full_weekdays, fill_value=0)
    weekly_defects_df = weekly_defects.reset_index()
    weekly_defects_df.columns = ["Date", "불량 LOT 수"]
    weekly_defects_df["Formatted_Date"] = (
        weekly_defects_df["Date"].dt.month.astype(str)
        + "월 "
        + weekly_defects_df["Date"].dt.day.astype(str)
        + "일"
    )

    fig = px.line(
        weekly_defects_df,
        x="Formatted_Date",
        y="불량 LOT 수",
        labels={"Formatted_Date": "날짜", "불량 LOT 수": "불량 LOT 수"},
        markers=True,
    )
    fig.update_layout(xaxis_title="날짜", yaxis_title="불량 LOT 수")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    pass

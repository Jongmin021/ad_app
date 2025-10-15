import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid

# 페이지 설정
st.set_page_config(page_title="리튬이온배터리 SoC 예측", layout="wide")

# 커스텀 CSS
st.markdown("""
<style>
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: 600;
        color: #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
    }
    
    /* Streamlit 기본 패딩 제거 */
    .block-container {
        padding-top: 2rem;
    }
    
    /* 이미지가 박스 밖으로 나가지 않도록 */
    .stImage {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stImage"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 데이터프레임 높이 제한 */
    .dataframe-container {
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# 타이틀
st.title("🔋 리튬이온배터리 SoC 예측 대시보드")

# 탭 생성
tab1, tab2 = st.tabs(["📊 모델 성능", "🚗 주행거리 예측"])

# 탭 1: 모델 성능
with tab1:
    st.header("모델링 예측")
    
    # 파일 업로드
    uploaded_file = st.file_uploader("모델 파일을 업로드하세요", type=['h5', 'pkl', 'pt'])
    
    # 파일이 업로드되고 특정 파일명인 경우에만 결과 표시
    if uploaded_file is not None and uploaded_file.name == 'Transformer_SOCpred.h5':
        st.success(f"✅ {uploaded_file.name} 파일이 업로드되었습니다.")
        
        # 이미지 파일 경로 설정
        r2_graph_path = "R2_TF.png"
        loss_graph_path = "loss_epoch_TF.png"
        test1_path = "Test1.png"
        test2_path = "Test2.png"
        test3_path = "Test3.png"
        test4_path = "Test4.png"
        test5_path = "Test5.png"
        test6_path = "Test6.png"
        
        # 좌우 5:5 분할
        col_left, col_right = st.columns([5, 5])
        
        # 왼쪽 열 (상하 5:5)
        with col_left:
            # 좌상단: R2 그래프
            st.markdown("### ☑️ R² 그래프")
            
            # R2 그래프 이미지
            if r2_graph_path:
                try:
                    img = Image.open(r2_graph_path)
                    # 이미지 크기를 50%로 축소
                    width, height = img.size
                    new_size = (int(width * 0.8), int(height * 0.8))
                    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    # 중앙 정렬을 위한 컨테이너
                    col_center = st.columns([1, 4, 1])
                    with col_center[1]:
                        st.image(img_resized, use_container_width=True)
                except Exception as e:
                    st.info("R² 그래프 이미지 파일 경로를 확인해주세요.")
            else:
                st.info("R² 그래프 이미지 파일 경로를 입력하세요.")
            
            # 좌하단: Loss vs Epoch 그래프
            st.markdown("### ☑️ Loss vs Epoch 그래프")
            
            # 설명 텍스트
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 15px;'>
                <p style='font-size: 25px; color: #333; margin: 0; font-weight: 500;'>
                    모델의 과적합 여부를 판단합니다.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Loss 그래프 이미지
            if loss_graph_path:
                try:
                    img = Image.open(loss_graph_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.info("Loss vs Epoch 그래프 이미지 파일 경로를 확인해주세요.")
            else:
                st.info("Loss vs Epoch 그래프 이미지 파일 경로를 입력하세요.")
        
        # 오른쪽 열 (상하 3:7)
        with col_right:
            
            st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

            # 우상단: 모델 성능 지표 (30% 높이)
            st.markdown("### ☑️ 모델 성능 지표")
            
            # 1x3 메트릭 배치
            metric_cols = st.columns(3)
            
            metrics = [
                ("R²", "0.999989"),
                ("MAE", "0.000720"),
                ("RMSE", "0.001020")
            ]
            
            for col, (metric_name, metric_value) in zip(metric_cols, metrics):
                with col:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px;'>
                        <p style='font-size: 30px; color: #666; margin: 0; font-weight: 500;'>{metric_name}</p>
                        <p style='font-size: 50px; font-weight: bold; color: #000; margin: 5px 0;'>{metric_value}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 우상단과 우하단 사이 간격
            st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)
            
            # 우하단: 테스트 결과 - SOC Profile (70% 높이)
            st.markdown("### ☑️ 테스트 결과 - SoC Profile")
            
            # 드롭다운 메뉴 - 크기 조정
            st.markdown("""
            <style>
                div[data-baseweb="select"] > div {
                    font-size: 20px !important;
                    font-weight: bold !important;
                    height: 60px !important;
                }
                div[data-baseweb="select"] {
                    width: 25% !important;
                    min-width: 150px !important;
                }
                .stSelectbox label {
                    font-size: 20px !important;
                    font-weight: 600 !important;
                    color: #333 !important;
                }
                /* 드롭다운 옵션 텍스트 스타일 */
                div[role="listbox"] div {
                    font-size: 18px !important;
                    font-weight: 600 !important;
                }
            </style>
            """, unsafe_allow_html=True)
            
            test_option = st.selectbox(
                "테스트 선택",
                ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"],
                index=0,
                key="test_selector"
            )
            
            # 테스트별 이미지 매핑
            test_images = {
                "Test 1": test1_path,
                "Test 2": test2_path,
                "Test 3": test3_path,
                "Test 4": test4_path,
                "Test 5": test5_path,
                "Test 6": test6_path
            }
            
            # 선택된 테스트 이미지 표시
            selected_image_path = test_images[test_option]
            
            if selected_image_path:
                try:
                    img = Image.open(selected_image_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.info(f"{test_option} 이미지 파일 경로를 확인해주세요.")
            else:
                st.info(f"{test_option} 이미지 파일 경로를 입력하세요.")
    
    elif uploaded_file is not None:
        st.warning("⚠️ 지원되는 모델 파일이 아닙니다. 'Transformer_SOCpred.h5' 파일을 업로드해주세요.")
    else:
        st.info("📁 모델 파일을 업로드하여 시작하세요.")

# 탭 2: 주행거리 예측
with tab2:
    st.header("주행거리 예측")
    
    # 상단: 데이터 업로드
    st.markdown("### ☑️ 데이터 업로드")
    drive_csv = st.file_uploader("주행 데이터 CSV 파일을 업로드하세요", type=['csv'], key="drive_csv_uploader")
    
    if drive_csv is not None:
        try:
            # CSV 파일 읽기
            df_full = pd.read_csv(drive_csv)
            
            # 각 온도 그룹별로 주행거리 계산
            df_full['Distance_km'] = 0.0
            
            for temp in df_full['ambient_temp'].unique():
                mask = df_full['ambient_temp'] == temp
                df_temp = df_full[mask].copy()
                
                # Speed_kmh를 m/s로 변환
                speed_ms = df_temp['Speed_kmh'].values / 3.6
                time_vals = df_temp['Time'].values
                
                # 각 그룹에서 시간 차이 계산 (dt)
                dt = np.diff(time_vals, prepend=time_vals[0])
                
                # 누적 거리 계산 (m 단위)
                distance_m = np.cumsum(speed_ms * dt)
                
                # km 단위로 변환하여 할당
                df_full.loc[mask, 'Distance_km'] = distance_m / 1000
            
            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
            
            # 하단: 1:1:2 비율로 분할
            col_left, col_center, col_right = st.columns([1, 1, 2])
            
            # 좌측: 온도 선택 및 데이터 표시
            with col_left:
                # 온도 드롭다운 (상단 20%)
                st.markdown("### ☑️ 온도 설정")
                temp_option = st.selectbox(
                    "온도 선택",
                    [40, 25, 10, 0, -10, -20],
                    index=1,  # 기본값 25℃
                    key="temp_selector",
                    format_func=lambda x: f"{x}℃"
                )
                
                st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
                
                # 선택된 온도에 해당하는 데이터 필터링 및 1800개 행만 선택
                df_filtered = df_full[df_full['ambient_temp'] == temp_option].head(1800).copy()
                
                # 데이터 표시 (하단 80%)
                st.markdown("### ☑️ 데이터")
                display_cols = ['Time', 'SOC_pred', 'Temperature', 'Speed_kmh', 'Distance_km']
                df_display = df_filtered[display_cols].copy()
                df_display.columns = ['Time', 'SOC_pred', 'Temperature', 'Speed', 'Distance']
                
                st.dataframe(df_display, height=500, use_container_width=True)
            
            # 중앙: 주행거리 정보
            with col_center:
                # 마지막 600개 데이터로 통계 계산
                df_last_600 = df_filtered.tail(600)
                
                # 통계값 계산
                soc_change = df_last_600['SOC_pred'].iloc[0] - df_last_600['SOC_pred'].iloc[-1]
                temp_range = f"{df_last_600['Temperature'].min():.1f}℃ ~ {df_last_600['Temperature'].max():.1f}℃"
                distance_10min = df_last_600['Distance_km'].iloc[-1] - df_last_600['Distance_km'].iloc[0]
                
                # 예상 잔여 주행거리 계산
                distance_per_001soc = distance_10min / soc_change if soc_change > 0 else 0
                remaining_soc = df_filtered['SOC_pred'].iloc[-1] - 0.05
                remaining_distance = remaining_soc * distance_per_001soc if remaining_soc > 0 else 0
                
                # 예상 주행 가능 시간 계산
                soc_change_per_min = soc_change / 10 if soc_change > 0 else 0
                remaining_time = remaining_soc / soc_change_per_min if soc_change_per_min > 0 else 0
                
                # 상단: 예상 주행거리 (30%)
                st.markdown("### ☑️ 예상 주행 정보")
                
                st.markdown(f"""
                <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 15px;'>
                    <p style='font-size: 18px; color: #1976d2; margin: 0; font-weight: 600;'>예상 잔여 주행거리</p>
                    <p style='font-size: 40px; font-weight: bold; color: #0d47a1; margin: 10px 0 0 0;'>{remaining_distance:.2f} km</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background-color: #f3e5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                    <p style='font-size: 18px; color: #7b1fa2; margin: 0; font-weight: 600;'>예상 주행 가능 시간</p>
                    <p style='font-size: 40px; font-weight: bold; color: #4a148c; margin: 10px 0 0 0;'>{remaining_time:.1f} 분</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 하단: 10분간 통계 (70%)
                st.markdown("### ☑️ 10분간 주행 통계")
                
                stats_data = [
                    ("SoC 변화량", f"{soc_change*100:.2f}%"),
                    ("온도 범위", temp_range),
                    ("주행거리", f"{distance_10min:.2f} km")
                ]
                
                for stat_name, stat_value in stats_data:
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                        <p style='font-size: 18px; color: #666; margin: 0; font-weight: 500;'>{stat_name}</p>
                        <p style='font-size: 28px; font-weight: bold; color: #000; margin: 5px 0 0 0;'>{stat_value}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 우측: 변수 프로파일 (3행 1열)
            with col_right:
                st.markdown("### ☑️ 주행 프로파일")
                
                # 1800개 데이터를 일반(1200개)과 하이라이트(600개)로 분리
                df_normal = df_filtered.iloc[:1200]
                df_highlight = df_filtered.iloc[1200:]
                
                # SOC 프로파일
                fig_soc = go.Figure()
                
                # 일반 구간
                fig_soc.add_trace(go.Scatter(
                    x=df_normal['Time'],
                    y=df_normal['SOC_pred'],
                    mode='lines',
                    name='SoC',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # 하이라이트 구간
                fig_soc.add_trace(go.Scatter(
                    x=df_highlight['Time'],
                    y=df_highlight['SOC_pred'],
                    mode='lines',
                    name='SoC (10분)',
                    line=dict(color='#ff7f0e', width=3)
                ))
                
                fig_soc.update_layout(
                    title=dict(text="SoC Profile", font=dict(size=22, weight='bold')),
                    xaxis_title="Time (s)",
                    yaxis_title="SoC",
                    height=250,
                    showlegend=False,
                    margin=dict(l=40, r=20, t=50, b=40)
                )
                
                st.plotly_chart(fig_soc, use_container_width=True)
                
                # Speed 프로파일
                fig_speed = go.Figure()
                
                # 일반 구간
                fig_speed.add_trace(go.Scatter(
                    x=df_normal['Time'],
                    y=df_normal['Speed_kmh'],
                    mode='lines',
                    name='Speed',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # 하이라이트 구간
                fig_speed.add_trace(go.Scatter(
                    x=df_highlight['Time'],
                    y=df_highlight['Speed_kmh'],
                    mode='lines',
                    name='Speed (10분)',
                    line=dict(color='#ff7f0e', width=3)
                ))
                
                fig_speed.update_layout(
                    title=dict(text="Speed Profile", font=dict(size=22, weight='bold')),
                    xaxis_title="Time (s)",
                    yaxis_title="Speed (km/h)",
                    height=250,
                    showlegend=False,
                    margin=dict(l=40, r=20, t=50, b=40)
                )
                
                st.plotly_chart(fig_speed, use_container_width=True)
                
                # 주행거리 프로파일
                fig_dist = go.Figure()
                
                # 일반 구간
                fig_dist.add_trace(go.Scatter(
                    x=df_normal['Time'],
                    y=df_normal['Distance_km'],
                    mode='lines',
                    name='Distance',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # 하이라이트 구간
                fig_dist.add_trace(go.Scatter(
                    x=df_highlight['Time'],
                    y=df_highlight['Distance_km'],
                    mode='lines',
                    name='Distance (10분)',
                    line=dict(color='#ff7f0e', width=3)
                ))
                
                fig_dist.update_layout(
                    title=dict(text="Distance Profile", font=dict(size=22, weight='bold')),
                    xaxis_title="Time (s)",
                    yaxis_title="Distance (km)",
                    height=250,
                    showlegend=False,
                    margin=dict(l=40, r=20, t=50, b=40)
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
        except Exception as e:
            st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
    else:

        st.info("📁 주행 데이터 CSV 파일을 업로드하여 시작하세요.")

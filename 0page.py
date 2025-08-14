# 0main.py
import streamlit as st
from datetime import date, timedelta
import importlib

st.set_page_config(page_title="통합 대시보드", layout="wide")
st.title("품질 예측 대시보드")


def import_show(modname: str):
    mod = importlib.import_module(modname)
    if not hasattr(mod, "show"):
        raise AttributeError(
            f"모듈 '{modname}'에 show()가 없습니다. def show(...): 를 정의하세요."
        )
    return getattr(mod, "show")


# 파일명은 겹치지 않게! (권장: kpi_page.py, utility_page.py, overlay_page.py)
show_kpi = import_show("page")  # page.py 에 def show(...)
show_utility = import_show("ppage")  # ppage.py 에 def show(...)
show_dash = import_show("pppage")  # pppage.py 에 def show(...)

# 공통 파라미터(원하면 전달)
selected_date = date.today()
date_range = (date.today() - timedelta(days=7), date.today())

tab1, tab2, tab3 = st.tabs(
    ["📈 KPI 대시보드", "⚙️ 유틸리티 실시간 모니터링", "🧪 불량 분석 대시보드"]
)

with tab1:
    show_kpi(selected_date=selected_date, date_range=date_range)

with tab2:
    show_utility(selected_date=selected_date, date_range=date_range)

with tab3:
    show_dash(selected_date=selected_date, date_range=date_range)

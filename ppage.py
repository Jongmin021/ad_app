import streamlit as st


def show(selected_date=None, date_range=None):
    st.title("유틸리티 실시간 모니터링")

    # app.py
    import base64, io, time
    from collections import deque
    from pathlib import Path
    import pandas as pd

    # 사이드바
    DEFAULT_IMG = "/Users/t2023-m0056/Downloads/KakaoTalk_Photo_2025-08-14-02-25-08.png"
    DEFAULT_CSV = "/Users/t2023-m0056/Desktop/파일/df_sensor.csv"
    st.sidebar.header("경로 설정")
    panel_img_path = st.sidebar.text_input("패널 이미지 경로", DEFAULT_IMG)
    df_path = st.sidebar.text_input("CSV 경로", DEFAULT_CSV)
    panel_max_w = st.sidebar.slider("패널 최대 너비(px)", 1200, 2600, 1400, 100)
    log_size = st.sidebar.slider("로그 표시 개수", 5, 20, 10, 1)

    # -------------------- 글로벌 스타일 --------------------
    st.markdown(
        """
    <style>
    /* 상단 KPI 카드 */
    .kpi{
    position:relative;
    display:flex; align-items:center; justify-content:center; flex-direction:column;
    height:120px; border-radius:14px; background:#ffffff;
    box-shadow:0 4px 12px rgba(0,0,0,.14);
    text-align:center;
    transition: transform .2s ease;
    }
    /* 레이블: 상단 30% 지점 */
    .kpi .label{
    position:absolute; top:30%; left:50%; transform:translate(-50%,-50%);
    font-size:13px; color:#5f6b7a; font-weight:500; margin:0;
    }
    /* 값: 세로 2/3(66%) 지점 + 1.5배 */
    .kpi .val{
    position:absolute; top:66%; left:50%; transform:translate(-50%,-50%);
    font-size:40px; font-weight:650; color:#0f2233; line-height:1;
    }

    /* KPI 애니메이션 (새 행 들어올 때 딱 한 번 튕기고 글로우) */
    @keyframes kpiPop { 0%{transform:scale(.98);} 100%{transform:scale(1);} }
    @keyframes kpiGlow {
    0%{ box-shadow:0 4px 12px rgba(0,0,0,.14); }
    40%{ box-shadow:0 0 0 rgba(0,0,0,0), 0 0 24px rgba(59,130,246,.35); }
    100%{ box-shadow:0 4px 12px rgba(0,0,0,.14); }
    }
    .kpi.changed{ animation: kpiPop 260ms ease-out, kpiGlow 900ms ease-out; }

    .kpi-wrap [data-testid="stHorizontalBlock"]{ gap:12px; } /* columns spacing */

    /* 공통 섹션 타이틀(그래프/테이블 동일) */
    .section-title{
    font-size:30px; font-weight:1000; margin:0 4px 6px; color:#0f2233;
    }

    /* 그래프 박스 여백 최소화 */
    [data-testid="stVegaLiteChart"]{ margin-top:0; }

    /* 테이블 */
    .table-wrap{ margin-top:8px; padding:14px 16px; border-radius:12px;
    background:linear-gradient(180deg, rgba(38,100,142,.10), rgba(38,100,142,.06));
    box-shadow:0 8px 18px rgba(0,0,0,.05);
    }
    .table-log{
    width:100%; table-layout:fixed; border-collapse:separate; border-spacing:0;
    }
    .table-log th, .table-log td{
    text-align:center; vertical-align:middle; padding:12px 10px; height:52px;
    border-bottom:1px solid rgba(0,0,0,.10);
    }
    /* 헤더: 조금 더 큼 */
    .table-log thead th{
    color:#0f2233; font-size:19px; font-weight:1000; background:rgba(255,255,255,.6);
    }
    /* 바디: 살짝 얇게 */
    .table-log tbody td{
    color:#0f2233; font-weight:500; background:#fff;
    }

    /* 이미지-그래프 사이 공백 제거용 */
    .panel-wrap{ max-width:100%; margin:0 auto 0; }  /* bottom=0 */

    /* ▶ 로그 신규 행 애니메이션 */
    @keyframes rowPop { 0%{transform:translateY(-6px); opacity:.0;} 100%{transform:translateY(0); opacity:1;} }
    @keyframes rowFlash { 0%{background:#e6fffa;} 60%{background:#ffffff;} 100%{background:#ffffff;} }
    .table-log tbody tr.new { animation: rowPop 300ms ease-out; }
    .table-log tbody tr.new td { animation: rowFlash 1200ms ease-out forwards; }

    /* 우상단 토글 */
    .topbar{ display:flex; justify-content:flex-end; align-items:center; margin-bottom:6px; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # -------------------- 데이터 로드 --------------------
    p_csv = Path(df_path)
    if not p_csv.exists():
        st.error(f"CSV 파일 없음: {p_csv}")
        st.stop()

    try:
        df = pd.read_csv(p_csv)
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")
        st.stop()

    required = {"DateTime", "Lot", "pH", "Temp", "Current", "Voltage"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"누락 컬럼: {missing}")
        st.stop()

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    df["row_id"] = df.index
    if df.empty:
        st.error("유효한 DateTime 데이터 없음")
        st.stop()

    # 시뮬레이션 시작 구간(마지막 300행 중 시작)
    start_idx = max(len(df) - 300, 0)
    sim_start_dt = df.loc[start_idx, "DateTime"]
    last_dt = df["DateTime"].iloc[-1]

    # -------------------- 세션 상태 --------------------
    if "sim_t0" not in st.session_state:
        st.session_state.sim_t0 = time.monotonic()
        st.session_state.sim_base_dt = sim_start_dt
        st.session_state.cursor = start_idx
        st.session_state.last_cursor = start_idx - 1
        st.session_state.log_rows = deque(maxlen=log_size)
        st.session_state.last_kpi_cursor = start_idx - 1

    if st.session_state.log_rows.maxlen != log_size:
        st.session_state.log_rows = deque(
            list(st.session_state.log_rows)[:log_size], maxlen=log_size
        )

    if "paused" not in st.session_state:
        st.session_state.paused = False

    # 자동 갱신
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=1000, key="tick")

    # 진행 커서
    if not st.session_state.paused:
        elapsed_s = time.monotonic() - st.session_state.sim_t0
        target_dt = min(
            st.session_state.sim_base_dt + pd.to_timedelta(elapsed_s, unit="s"), last_dt
        )
        idx = int(df["DateTime"].searchsorted(target_dt, side="right") - 1)
        st.session_state.cursor = max(idx, start_idx)

    cur = df.loc[st.session_state.cursor]
    cur_dt = cur["DateTime"]

    # 오늘 기준 보조
    today = cur_dt.date()
    df_today = df[df["DateTime"].dt.date == today].copy()
    df_today["DayIndex"] = range(1, len(df_today) + 1)
    day_index_map = dict(zip(df_today["row_id"], df_today["DayIndex"]))
    cur_day_index = int(day_index_map.get(int(cur["row_id"]), 0))

    # 오늘 공정 시작시간
    day_first = df_today["DateTime"].min()
    elapsed_td = cur_dt - day_first

    def fmt_td(td: pd.Timedelta) -> str:
        s = int(td.total_seconds())
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02}:{m:02}:{s:02}"

    # -------------------- 우상단 토글 --------------------
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    tog_l, tog_r = st.columns([8, 1])
    st.session_state.paused = tog_r.toggle(
        "모니터링 중지", value=st.session_state.paused
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- 상단 KPI (새 행 들어오면 애니메이션) --------------------
    animate_kpi = st.session_state.cursor > st.session_state.last_kpi_cursor

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    def kpi(col, label, value, changed=False):
        cls = "kpi changed" if changed else "kpi"
        col.markdown(
            f"""
        <div class="{cls}">
        <div class="label">{label}</div>
        <div class="val">{value}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with k1:
        kpi(k1, "현재 LOT", str(cur["Lot"]), changed=animate_kpi)
    with k2:
        kpi(k2, "현재 시각", cur_dt.strftime("%H:%M:%S"), changed=animate_kpi)
    with k3:
        kpi(k3, "공정 시작 시간", day_first.strftime("%H:%M:%S"), changed=animate_kpi)
    with k4:
        kpi(k4, "경과 시간", fmt_td(elapsed_td), changed=animate_kpi)
    with k5:
        kpi(k5, "Index", cur_day_index, changed=animate_kpi)

    st.session_state.last_kpi_cursor = st.session_state.cursor
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # ---- SVG 패널 (점/라인 색 변경, 추가 형광초록 점) --------------------
    from PIL import Image
    import streamlit.components.v1 as components

    def panel_temp_ph_iv(
        img_path: str,
        row,
        box_w=260,
        box_h=110,
        crop_bottom_px=120,
        iframe_h=None,
        wrap_max_w=None,
    ):
        from PIL import Image
        import base64, io
        from pathlib import Path
        import streamlit.components.v1 as components

        p = Path(img_path)
        if not p.exists():
            st.info("패널 이미지 없음. SVG 패널 생략.")
            return

        im = Image.open(p).convert("RGBA")
        W, H = im.size
        Hc = max(1, H - int(crop_bottom_px))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64img = base64.b64encode(buf.getvalue()).decode()

        # 팔레트
        COL = {
            "card_fill": "#E8EDF3",
            "card_edge": "#00E5FF",  # 센서 카드/파랑 라인
            "station_fill_a": "#F3F6FA",
            "station_fill_b": "#E8EDF3",
            "station_edge": "#00FF66",  # 형광초록 테두리(유지)
            "text": "#0F172A",
            "wire": "#00E5FF",  # 연결선 파랑
            "dot_blue": "#00FF66",  # ▶ 형광파랑 점/라인 → 파란색으로
            "neon_green": "#00FF66",  # ▶ 추가로 찍을 형광초록 점
        }
        clamp = lambda v, lo, hi: max(lo, min(hi, v))

        SCALE = max(0.9, min(1.8, W / 1600))
        card_w = int(box_w * SCALE)
        card_h = int(box_h * SCALE)
        r = int(12 * SCALE)
        pad = int(16 * SCALE)
        led_r = int(8 * SCALE)
        wire_w = clamp(int(min(W, Hc) * 0.0042), 3, 6)

        # 변수명/값 폰트
        label_fs = clamp(int(card_h * 0.26), 12, 20)
        value_fs = clamp(int(card_h * 0.46), 26, 50)

        # ── 센서 카드: 왼쪽 세로 스택 ──
        left_pad = int(16 * SCALE)
        v_gap = int(18 * SCALE)
        first_y = clamp(int(Hc * 0.14), 0, Hc - 4 * card_h - 3 * v_gap)
        stack_x = left_pad
        cards = [
            {
                "x": stack_x,
                "y": first_y + (card_h + v_gap) * i,
                "label": lb,
                "val": f"{val:.2f}",
            }
            for i, (lb, val) in enumerate(
                [
                    ("🌡️ 온도(°C)", float(row["Temp"])),
                    ("🧪 pH", float(row["pH"])),
                    ("🔋 전류(A)", float(row["Current"])),
                    ("⚡ 전압(V)", float(row["Voltage"])),
                ]
            )
        ]

        # 파랑 앵커 점(변경 없음)
        anchor_ax, anchor_ay = 504, 376

        def elbow(ax, ay, bx, by):
            return f"{ax},{ay} {ax},{by} {bx},{by}"

        # 공정 라벨
        stations_top = [
            "탈지조",
            "산세조",
            "수세조",
            "전해탈지조",
            "수세조",
            "Innozinc 도금조",
        ]
        bottom_left_stack = ["건조기", "크로메이트/부동태화조"]
        bottom_right_row = ["수세조", "활성화조", "수세조"]

        # 공정 밴드 레이아웃
        stack_right = stack_x + card_w
        stack_gap = int(40 * SCALE)
        band_shift = int(28 * SCALE)
        belt_ratio = 0.72
        band_l_base = int((1 - belt_ratio) / 2 * W)
        band_l = max(band_l_base, stack_right + stack_gap) + band_shift
        band_r = W - band_l_base

        station_h = int(40 * SCALE)
        top_y = clamp(first_y - int(22 * SCALE) - station_h, 0, Hc - station_h)
        bot_y = clamp(Hc - station_h - int(120 * SCALE), 0, Hc - station_h)

        def label_width(lb: str) -> int:
            base = int(120 * SCALE)
            per = int(7 * SCALE)
            bw = base + per * len(lb)
            if "크로메이트" in lb:
                bw = max(bw + int(70 * SCALE), int(330 * SCALE))
            return clamp(bw, int(110 * SCALE), int(520 * SCALE))

        def layout_row(labels, y):
            n = len(labels)
            xs = (
                [int(band_l + i * (band_r - band_l) / (n - 1)) for i in range(n)]
                if n > 1
                else [max(band_l, W // 2)]
            )
            items = []
            for cx, lb in zip(xs, labels):
                bw = label_width(lb)
                left = clamp(cx - bw // 2, 0, W - bw)
                items.append({"left": left, "bw": bw, "y": y, "label": lb})
            return items

        def layout_bottom(left_stack, right_row, y):
            v_gap2 = int(10 * SCALE)
            down_shift = int(12 * SCALE)
            w0 = label_width(left_stack[0])
            w1 = label_width(left_stack[1])
            left_col = band_l
            items = [
                {
                    "left": left_col,
                    "bw": w0,
                    "y": y - (station_h + v_gap2) + down_shift,
                    "label": left_stack[0],
                },
                {
                    "left": left_col,
                    "bw": w1,
                    "y": y + down_shift,
                    "label": left_stack[1],
                },
            ]
            cur_left = left_col + max(w0, w1) + int(18 * SCALE)
            for lb in right_row:
                bw = label_width(lb)
                items.append({"left": cur_left, "bw": bw, "y": y, "label": lb})
                cur_left += bw + int(18 * SCALE)
            return items

        top_items = layout_row(stations_top, top_y)
        bot_items = layout_bottom(bottom_left_stack, bottom_right_row, bot_y)

        # ─ SVG ─
        if iframe_h is None:
            iframe_h = int(Hc + 180)
        svg = [
            f'<svg viewBox="0 0 {W} {Hc}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">'
        ]
        svg += [
            f"""
        <defs>
            <linearGradient id="stationGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%"  stop-color="{COL['station_fill_a']}"/><stop offset="100%" stop-color="{COL['station_fill_b']}"/>
            </linearGradient>
            <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="2.2" result="g"/><feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="neon-glow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="3" result="b1"/>
            <feFlood flood-color="{COL['card_edge']}" flood-opacity="0.9" result="c1"/>
            <feComposite in="c1" in2="b1" operator="in" result="s1"/>
            <feMerge><feMergeNode in="s1"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <image x="0" y="0" width="{W}" height="{Hc}" href="data:image/png;base64,{b64img}" preserveAspectRatio="xMidYMid meet"/>
        """
        ]
        # 파랑 앵커 점
        svg += [
            f'<circle cx="{anchor_ax}" cy="{anchor_ay}" r="{max(2,int(3*SCALE))}" fill="{COL["card_edge"]}" filter="url(#neon-glow)"/>'
        ]
        # 추가 파랑 점(고정)
        svg += [
            f'<circle cx="505" cy="320" r="{max(5,int(7*SCALE))}" fill="{COL["card_edge"]}" filter="url(#neon-glow)"/>'
        ]

        # 센서 연결선(파랑)
        for c in cards:
            cx = c["x"] + card_w / 2
            cy = c["y"] + card_h / 2
            svg += [
                f'<polyline filter="url(#neon-glow)" points="{elbow(anchor_ax, anchor_ay, cx, cy)}" '
                f'fill="none" stroke="{COL["wire"]}" stroke-width="{wire_w}" stroke-linecap="square" stroke-linejoin="miter" />'
            ]

        # 센서 카드
        for c in cards:
            x, y = c["x"], c["y"]
            label_y = y + int(card_h * 0.32)
            value_y = y + int(card_h * 0.76)
            svg += [
                f"""
            <g filter="url(#neon-glow)">
                <rect x="{x}" y="{y}" rx="{r}" ry="{r}" width="{card_w}" height="{card_h}"
                    fill="{COL['card_fill']}" stroke="{COL['card_edge']}" stroke-width="3"/>
                <text x="{x + card_w/2}" y="{label_y}" fill="{COL['text']}" font-size="{label_fs}" font-weight="600" text-anchor="middle">{c['label']}</text>
                <text x="{x + card_w/2}" y="{value_y}" fill="{COL['text']}" font-size="{value_fs}" font-weight="900" text-anchor="middle">{c['val']}</text>
            </g>
            """
            ]

        # 공정 박스
        def render_stations(items):
            parts = []
            for d in items:
                parts += [
                    f"""
                <g filter="url(#glow)">
                    <rect x="{d['left']}" y="{d['y']}" rx="{r-2}" ry="{r-2}" width="{d['bw']}" height="{station_h}"
                        fill="url(#stationGrad)" stroke="{COL['station_edge']}" stroke-width="2"/>
                    <g>
                    <circle cx="{d['left'] + d['bw'] - pad}" cy="{d['y'] + station_h/2}" r="{led_r}" fill="#22C55E"/>
                    <circle cx="{d['left'] + d['bw'] - pad}" cy="{d['y'] + station_h/2}" r="{led_r+2}" fill="#22C55E" opacity="0.45">
                        <animate attributeName="r" values="{led_r+2};{led_r+9};{led_r+2}" dur="1.2s" repeatCount="indefinite"/>
                        <animate attributeName="opacity" values="0.45;0;0.45" dur="1.2s" repeatCount="indefinite"/>
                    </circle>
                    </g>
                    <text x="{d['left'] + d['bw']/2}" y="{d['y'] + station_h*0.66}"
                        fill="{COL['text']}" font-size="{int(20*SCALE)}" font-weight="800" text-anchor="middle">{d['label']}</text>
                </g>
                """
                ]
            return parts

        svg += render_stations(top_items) + render_stations(bot_items)

        # ── 파란 점/라인 (기존 형광 'green_points' → 파란색으로 변경) ──
        OFFSET_Y = -60

        def rect_border_point(px, py, left, top, bw, h):
            right = left + bw
            bottom = top + h
            qx = clamp(px, left, right)
            qy = clamp(py, top, bottom)
            if px < left or px > right or py < top or py > bottom:
                return qx, qy
            candidates = [
                (abs(py - top), (px, top)),
                (abs(py - bottom), (px, bottom)),
                (abs(px - left), (left, py)),
                (abs(px - right), (right, py)),
            ]
            return min(candidates, key=lambda t: t[0])[1]

        def find_nth(items, label, n=0):
            k = 0
            for it in items:
                if it["label"] == label:
                    if k == n:
                        return it
                    k += 1
            for it in items:
                if it["label"] == label:
                    return it
            return None

        green_points_top = [
            ((393, 364), ("탈지조", 0)),
            ((552, 364), ("산세조", 0)),
            ((687, 273), ("수세조", 0)),  # 예외: 오프셋 제거
            ((721, 271), ("전해탈지조", 0)),  # 예외: 오프셋 제거
            ((770, 370), ("수세조", 1)),
            ((844, 370), ("Innozinc 도금조", 0)),
        ]
        green_points_bottom = [
            ((750, 460), ("수세조", 1)),
            ((700, 485), ("활성화조", 0)),
            ((629, 373), ("수세조", 0)),
            ((473, 485), ("건조기", 0)),
            ((443, 449), ("크로메이트/부동태화조", 0)),
        ]
        NO_OFFSET_TOP_POINTS = {(687, 273), (721, 271)}
        dot_r = max(1, int(2 * SCALE))
        stroke_w_g = max(2, int(3 * SCALE))

        # 윗줄(파란색)
        for (px, py), (lb, occ) in green_points_top:
            py_used = py if (px, py) in NO_OFFSET_TOP_POINTS else (py + OFFSET_Y)
            it = find_nth(top_items, lb, occ)
            if it:
                tx, ty = rect_border_point(
                    px, py_used, it["left"], it["y"], it["bw"], station_h
                )
                svg += [
                    f'<circle cx="{px}" cy="{py_used}" r="{dot_r}" fill="{COL["dot_blue"]}" filter="url(#glow)"/>',
                    f'<line x1="{px}" y1="{py_used}" x2="{tx}" y2="{ty}" stroke="{COL["dot_blue"]}" stroke-width="{stroke_w_g}" filter="url(#glow)" stroke-linecap="square" />',
                ]
        # 아랫줄(파란색)
        for (px, py), (lb, occ) in green_points_bottom:
            py_off = py + OFFSET_Y
            it = find_nth(bot_items, lb, occ)
            if it:
                tx, ty = rect_border_point(
                    px, py_off, it["left"], it["y"], it["bw"], station_h
                )
                svg += [
                    f'<circle cx="{px}" cy="{py_off}" r="{dot_r}" fill="{COL["dot_blue"]}" filter="url(#glow)"/>',
                    f'<line x1="{px}" y1="{py_off}" x2="{tx}" y2="{ty}" stroke="{COL["dot_blue"]}" stroke-width="{stroke_w_g}" filter="url(#glow)" stroke-linecap="square" />',
                ]

            # ---- (도우미) 같은 라벨이 여러 개면 X좌표가 가장 가까운 박스를 선택 ----

        def find_station_nearest_by_x(items, label, px):
            cands = [it for it in items if it["label"] == label]
            if not cands:
                return None
            # 박스 중심 x 와 포인트 px 간의 거리로 최단 선택
            return min(cands, key=lambda it: abs((it["left"] + it["bw"] / 2) - px))

        # ▶ 형광초록 점 4곳을 해당 공정 박스 "테두리"까지 형광초록 직선으로 연결 (루프 밖에서 단 한 번 실행)
        conn_specs = [
            {
                "pt": (480, 420),
                "row": "bottom",
                "label": "크로메이트/부동태화조",
            },  # 크로메이트
            {
                "pt": (600, 410),
                "row": "bottom",
                "label": "수세조",
            },  # 크로메이트 옆 수세조
            {"pt": (630, 320), "row": "top", "label": "수세조"},  # 산세조 옆 수세조
            {"pt": (670, 310), "row": "top", "label": "전해탈지조"},  # 전해탈지조
        ]

        for cs in conn_specs:
            px, py = cs["pt"]
            items = top_items if cs["row"] == "top" else bot_items
            it = find_station_nearest_by_x(items, cs["label"], px)
            if it:
                tx, ty = rect_border_point(
                    px, py, it["left"], it["y"], it["bw"], station_h
                )
                svg += [
                    f'<circle cx="{px}" cy="{py}" r="{max(3,int(4*SCALE))}" fill="{COL["neon_green"]}" filter="url(#glow)"/>',
                    f'<line x1="{px}" y1="{py}" x2="{tx}" y2="{ty}" stroke="{COL["neon_green"]}" stroke-width="{stroke_w_g}" filter="url(#glow)" stroke-linecap="square" />',
                ]

        svg += ["</svg>"]

        wrap_style = (
            f"width:100%; max-width:{wrap_max_w}px; margin:0 auto; padding:56px 0;"
            if wrap_max_w
            else "width:100vw; max-width:100vw; margin:0 calc(-50vw + 50%); padding:56px 0;"
        )
        components.html(
            f'<div class="panel-wrap" style="{wrap_style}">{"".join(svg)}</div>',
            height=iframe_h or int(Hc + 180),
            scrolling=False,
        )

    panel_temp_ph_iv(
        panel_img_path,
        cur,
        box_w=260,
        box_h=110,
        iframe_h=None,
        wrap_max_w=panel_max_w,  # ▶ 슬라이더 적용
    )

    # -------------------- 실시간 라인차트 (항상 최근 50개 슬라이딩) --------------------
    import altair as alt

    WINDOW = 50
    start_idx_w = max(st.session_state.cursor - (WINDOW - 1), 0)
    mask_range = (df.index >= start_idx_w) & (df.index <= st.session_state.cursor)

    st.markdown(
        '<div class="section-title">실시간 센서 데이터 트렌드</div>',
        unsafe_allow_html=True,
    )

    df_plot = (
        df.loc[mask_range, ["DateTime", "pH", "Temp", "Current", "Voltage"]]
        .dropna()
        .copy()
    )

    if not df_plot.empty:
        df_rt = df_plot.rename(
            columns={"Temp": "온도(°C)", "Current": "전류(A)", "Voltage": "전압(V)"}
        )
        tidy = df_rt.melt(["DateTime"], var_name="변수", value_name="값")

        domain = ["pH", "온도(°C)", "전류(A)", "전압(V)"]
        range_ = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
        sel = alt.selection_point(fields=["변수"], bind="legend")

        axis_kor = alt.Axis(
            tickCount=8,
            labelAngle=0,
            labelFontSize=14,
            title=None,
            labelExpr=(
                "'오후 ' + (((hours(datum.value)%12)==0)?12:(hours(datum.value)%12)) "
                "+ ' : ' + ((minutes(datum.value)<10)?('0'+minutes(datum.value)):minutes(datum.value))"
            ),
        )

        chart = (
            alt.Chart(tidy)
            .mark_line(interpolate="monotone")
            .encode(
                x=alt.X("DateTime:T", axis=axis_kor),
                y=alt.Y("값:Q", title=None),
                color=alt.Color(
                    "변수:N",
                    scale=alt.Scale(domain=domain, range=range_),
                    legend=alt.Legend(
                        title=None,
                        orient="top",
                        direction="horizontal",
                        labelFontSize=16,
                        symbolSize=200,
                        padding=6,
                    ),
                ),
                opacity=alt.condition(sel, alt.value(1.0), alt.value(0.18)),
                tooltip=[
                    alt.Tooltip("DateTime:T", title="Time"),
                    alt.Tooltip("변수:N", title="변수"),
                    alt.Tooltip("값:Q", title="값", format=".2f"),
                ],
            )
            .add_params(sel)
            .properties(height=340)
            .interactive()
        ).configure_view(strokeWidth=0)

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("표시할 데이터가 없습니다.")

    # -------------------- 최근 데이터 로그 --------------------
    def kor_time_full(ts) -> str:
        t = pd.to_datetime(ts)
        ap = "오전" if t.hour < 12 else "오후"
        h12 = t.hour % 12 or 12
        return f"{ap} {h12}:{t.minute:02d}:{t.second:02d}"

    def make_row_by_rowid(row_id: int) -> dict:
        sr = df.loc[row_id]
        return {
            "_row_id": int(sr["row_id"]),
            "Index": int(day_index_map.get(sr["row_id"], 0)),
            "LOT": str(sr["Lot"]),
            "날짜 (Date)": sr["DateTime"].date().isoformat(),
            "시간 (Time)": kor_time_full(sr["DateTime"]),
            "pH": f"{float(sr['pH']):.1f}",
            "온도 (Temp)": f"{float(sr['Temp']):.1f}°C",
            "전류 (Current)": f"{float(sr['Current']):.1f}A",
            "전압 (Voltage)": f"{float(sr['Voltage']):.1f}V",
        }

    # 초기 시드
    if len(st.session_state.log_rows) == 0:
        today_upto = df_today[df_today["DateTime"] <= cur_dt]["row_id"].tolist()
        seed_ids = today_upto[-log_size:] if len(today_upto) >= log_size else today_upto
        for rid in reversed(seed_ids):
            st.session_state.log_rows.appendleft(make_row_by_rowid(rid))
        st.session_state.last_cursor = st.session_state.cursor

    # 업데이트(최신 위로, 개수 고정)
    if st.session_state.cursor > st.session_state.last_cursor:
        new_top_id = None
        for i in range(st.session_state.last_cursor + 1, st.session_state.cursor + 1):
            if df.loc[i, "DateTime"].date() == today:
                item = make_row_by_rowid(i)
                st.session_state.log_rows.appendleft(item)
                new_top_id = item["_row_id"]
        st.session_state.last_cursor = st.session_state.cursor
        st.session_state.just_added_row_id = new_top_id  # ▶ 새로 들어온 맨 윗줄 키

    rows = list(st.session_state.log_rows)[:log_size]
    animate_row_id = st.session_state.pop("just_added_row_id", None)

    headers = [
        "Index",
        "LOT",
        "날짜 (Date)",
        "시간 (Time)",
        "pH",
        "온도 (Temp)",
        "전류 (Current)",
        "전압 (Voltage)",
    ]
    col_widths = ["8%", "18%", "14%", "14%", "9%", "12%", "12%", "13%"]

    table_html = [
        '<div class="table-wrap">',
        '<div class="section-title">📋 최근 데이터 로그</div>',
        '<table class="table-log"><colgroup>',
    ]
    for w in col_widths:
        table_html.append(f'<col style="width:{w}">')
    table_html.append("</colgroup><thead><tr>")
    for h in headers:
        table_html.append(f"<th>{h}</th>")
    table_html.append("</tr></thead><tbody>")

    for idx, r in enumerate(rows):
        tr_cls = (
            ' class="new"' if idx == 0 and r.get("_row_id") == animate_row_id else ""
        )
        table_html.append(
            f"<tr{tr_cls}>" + "".join(f"<td>{r[h]}</td>" for h in headers) + "</tr>"
        )

    table_html.append("</tbody></table></div>")
    st.markdown("".join(table_html), unsafe_allow_html=True)


if __name__ == "__main__":
    pass

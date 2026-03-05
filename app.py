import streamlit as st
import yfinance as yf
import pandas as pd
# pandas_ta → numba 의존성으로 Python 3.13 미지원, 순수 pandas로 대체
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import numpy as np
from scipy.signal import argrelextrema
from google import genai
import os
from dotenv import load_dotenv
from notion_client import Client
import json

# --- 환경 변수 로드 및 Gemini 설정 ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if api_key:
    client = genai.Client(api_key=api_key)
else:
    st.sidebar.error("Gemini API Key가 .env 파일에 설정되어 있지 않습니다.")

# --- Notion 설정 ---
notion_token = os.getenv("NOTION_TOKEN")
notion_db_id = os.getenv("NOTION_DATABASE_ID")

if notion_token and notion_db_id:
    notion = Client(auth=notion_token)
else:
    notion = None

# --- 페이지 설정 ---
st.set_page_config(
    page_title="QuantumBrief - Pro Analyst Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded" # 챗봇을 위해 시작 시 서랍장 열어둠
)

# --- 커스텀 스타일링 ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0B0E14;
        color: #E2E8F0;
    }
    .metric-card {
        background-color: #151921;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #2D3748;
        box-shadow: 0 14px 28px rgba(0,0,0,0.5);
        margin-bottom: 24px;
    }
    .ticker-title {
        font-size: 30px;
        font-weight: 900;
        color: #F7FAFC;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    .pattern-label {
        background: rgba(49, 130, 206, 0.15);
        color: #63B3ED;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 15px;
        border: 1px solid rgba(99, 179, 237, 0.3);
    }
    /* RSI Highlighter */
    .rsi-oversold { background-color: #2F855A; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .rsi-overbought { background-color: #C53030; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .rsi-neutral { color: #A0AEC0; }
    
    [data-testid="stMetricValue"] { font-size: 34px !important; font-weight: 800 !important; }
    .main-header {
        font-size: 48px; font-weight: 950; margin-bottom: 10px;
        background: linear-gradient(135deg, #FFFFFF 0%, #718096 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- RSI 계산 (pandas_ta 대체) ---
def calc_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- 하모닉 패턴 분석 알고리즘 ---
def detect_patterns(df):
    if len(df) < 40: return "Insufficient Data", None
    
    n = 5 
    df['min'] = df['Close'].iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]
    df['max'] = df['Close'].iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]
    
    points = df.dropna(subset=['min', 'max'], how='all')
    if len(points) < 5: return "No Pattern", None
    
    last_5 = points.tail(5)
    p_vals = last_5['Close'].values
    p_idx = last_5.index
    X, A, B, C, D = p_vals
    
    # Bullish/Bearish 판별
    is_bullish = X < A and B < A and B > X and C > B and C < A and D < C
    is_bearish = X > A and B > A and B < X and C < B and C > A and D > C

    # 비율 계산
    AB_XA = abs(B-A) / abs(A-X)
    CD_AB = abs(D-C) / abs(B-A) if abs(B-A) != 0 else 0
    AD_XA = abs(D-X) / abs(A-X)
    
    pattern_type = "Scanning"
    if 0.58 < AB_XA < 0.65: pattern_type = "Gartley"
    elif 0.38 < AB_XA < 0.52: pattern_type = "Bat"
    elif 0.75 < AB_XA < 0.82: pattern_type = "Butterfly"
    else: pattern_type = "Complex Structure"

    direction = "(Bullish 🔼)" if is_bullish else ("(Bearish 🔽)" if is_bearish else "")
    abcd_status = "AB=CD OK" if 0.88 < CD_AB < 1.12 else f"AB=CD ratio {CD_AB:.2f}"
    
    label = f"{pattern_type} {direction} | {abcd_status}"
    return label, last_5

# --- 통합 데이터 로드 ---
@st.cache_data(ttl=600)
def fetch_all_assets(tickers):
    data = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="6mo")
            if df.empty: continue
            
            df['RSI'] = calc_rsi(df['Close'], length=14)
            pat_label, pat_points = detect_patterns(df.copy())
            
            # Fetch name (Optional fallback)
            name = t
            try:
                # Use cached info if available to avoid extra requests
                name = stock.info.get('longName') or stock.info.get('shortName') or t
            except:
                pass

            data[t] = {
                'name': name,
                'df': df,
                'price': df['Close'].iloc[-1],
                'prev': df['Close'].iloc[-2],
                'vol': df['Volume'].iloc[-1],
                'rsi': df['RSI'].iloc[-1],
                'pattern_label': pat_label,
                'pattern_points': pat_points
            }
        except: continue
    return data

# --- 대시보드 메인 ---
st.markdown('<p class="main-header">💎 QuantumBrief Pro</p>', unsafe_allow_html=True)
st.caption(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Technical: Fibonacci Harmonic Ratios (Simplified)")

# --- 자산 관리 기능 ---
ASSETS_FILE = "assets.json"

def load_assets():
    try:
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # 기본값 반환 및 파일 생성
        default_assets = {
            "macro_ids": ["^TNX"],
            "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "us_stocks": ["IONQ", "PLTR", "NVDA", "TSLA", "FIG", "GOOGL", "LEU", "COHR", "ASTS", "TEM"],
            "kr_stocks": ["017670.KS", "128940.KS", "100790.KQ", "006800.KS", "380550.KQ", "036930.KQ"]
        }
        save_assets(default_assets)
        return default_assets

def save_assets(assets):
    with open(ASSETS_FILE, "w", encoding="utf-8") as f:
        json.dump(assets, f, indent=2, ensure_ascii=False)

# 자산 로드
assets_data = load_assets()
macro_ids = assets_data.get("macro_ids", [])
crypto = assets_data.get("crypto", [])
us_stocks = assets_data.get("us_stocks", [])
kr_stocks = assets_data.get("kr_stocks", [])
display_names = assets_data.get("display_names", {})
all_assets = macro_ids + crypto + us_stocks + kr_stocks

# 데이터 먼저 로드 (챗봇에서 사용하기 위해)
data_store = fetch_all_assets(all_assets)

# --- 10Y-3M 장단기 금리차 스프레드 계산 (가상 자산 추가) ---
if "^TNX" in data_store and "^IRX" in data_store:
    try:
        tnx = data_store["^TNX"]
        irx = data_store["^IRX"]
        # 두 시리즈를 날짜 기준으로 정렬 후 공통 인덱스로 스프레드 계산
        spread_close = tnx['df']['Close'].subtract(irx['df']['Close'], fill_value=None).dropna()
        spread_df = tnx['df'].copy()
        spread_df['Close'] = spread_close
        spread_df['Open']  = tnx['df']['Open'] - irx['df']['Open'].reindex(tnx['df'].index, method='nearest')
        spread_df['High']  = spread_df['Close']
        spread_df['Low']   = spread_df['Close']
        spread_df['RSI']   = calc_rsi(spread_df['Close'], length=14)
        spread_price = spread_close.iloc[-1]
        spread_prev  = spread_close.iloc[-2]
        pat_label, pat_points = detect_patterns(spread_df.copy())
        data_store["SPREAD_10Y2Y"] = {
            'name': '10Y-3M Spread',
            'df': spread_df,
            'price': spread_price,
            'prev': spread_prev,
            'vol': 0,
            'rsi': spread_df['RSI'].iloc[-1],
            'pattern_label': pat_label,
            'pattern_points': pat_points,
        }
    except Exception as e:
        pass  # 스프레드 계산 실패 시 무시

# 화면에 표시할 매크로 목록: ^IRX는 계산용이므로 제외, SPREAD_10Y2Y 추가
macro_display = [t for t in macro_ids if t != "^IRX"]
if "SPREAD_10Y2Y" in data_store:
    macro_display.append("SPREAD_10Y2Y")

# --- Q&A 저장 함수 ---
def save_qa_to_file(question, answer):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    dir_path = "java/answer"
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{date_str}.md")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"## [{time_str}] 질문\n")
        f.write(f"{question}\n\n")
        f.write(f"### 답변\n")
        f.write(f"{answer}\n\n")
        f.write("---\n\n")
    
    # --- Notion에 추가 저장 ---
    if notion:
        try:
            notion.pages.create(
                parent={"database_id": notion_db_id},
                properties={
                    "주식 분석": {"title": [{"text": {"content": question[:100] + "..." if len(question) > 100 else question}}]},
                    "날짜": {"date": {"start": datetime.now().isoformat()}},
                },
                children=[
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "질문"}}] }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": question}}] }
                    },
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "답변"}}] }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {"rich_text": [{"type": "text", "text": {"content": answer[:2000]}}] } # Notion 2000자 제한 대응
                    }
                ]
            )
        except Exception as e:
            st.error(f"Notion 저장 오류: {e}")


# --- 사이드바: Gemini 주식 챗봇 ---
with st.sidebar:
    st.markdown("### 🤖 Quantum Sidekick")
    st.markdown("현재 대시보드 데이터를 기반으로 질문에 답변하며, 모든 대화는 `java/answer` 폴더에 저장됩니다.")

    # --- Gemini 모델 선택 ---
    GEMINI_MODELS = {
        "⚡ gemini-flash-latest": "gemini-flash-latest",
        "✨ gemini-2.5-flash": "gemini-2.5-flash",
        "🚀 gemini-3-flash-preview": "gemini-3-flash-preview",
        "💨 gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "🔬 gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
    }
    selected_model_label = st.selectbox(
        "🧠 Gemini 모델 선택",
        options=list(GEMINI_MODELS.keys()),
        index=0,
        key="gemini_model_select"
    )
    selected_model = GEMINI_MODELS[selected_model_label]

    st.divider()

    # 채팅 기록 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 메시지 표시 컨테이너
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 사용자 입력 (사이드바 하단에 고정됨)
    if prompt := st.chat_input("이 종목들의 패턴에 대해 물어보세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # UI 즉시 업데이트를 위해 메시지 표시
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Gemini 답변 생성
            with st.chat_message("assistant"):
                if api_key:
                    try:
                        # 현재 데이터 컨텍스트 생성 (종목명 우선 표시)
                        context = "현재 분석 중인 종목 데이터:\n"
                        if data_store:
                            for tid, stats in data_store.items():
                                # display_names(사용자 지정) → yfinance name → ticker 코드 순으로 fallback
                                stock_name = display_names.get(tid) or stats.get('name') or tid
                                context += f"- {stock_name} ({tid}): 현재가 {stats['price']:.2f}, RSI {stats['rsi']:.1f}, 패턴 {stats['pattern_label']}\n"
                        else:
                            context += "데이터 로딩 중...\n"

                        system_prompt = f"너는 20년 경력의 시니어 퀀트 애널리스트야. 다음 데이터를 참고해서 사용자의 질문에 전문적이고 친절하게 답변해줘.\n\n{context}"

                        # 채팅 히스토리 포함 전송 (타이머 포함)
                        chat = client.chats.create(model=selected_model)
                        full_prompt = f"{system_prompt}\n\n사용자 질문: {prompt}"

                        start_time = time.time()
                        with st.status(f"⏳ {selected_model_label} 응답 생성 중...", expanded=False) as status:
                            timer_placeholder = st.empty()
                            # 실시간 타이머 업데이트 (폴링 방식)
                            import threading
                            response_holder = [None]
                            error_holder = [None]

                            def call_api():
                                try:
                                    response_holder[0] = chat.send_message(message=full_prompt)
                                except Exception as ex:
                                    error_holder[0] = ex

                            thread = threading.Thread(target=call_api)
                            thread.start()

                            while thread.is_alive():
                                elapsed = time.time() - start_time
                                timer_placeholder.markdown(f"⏱️ 경과 시간: **{elapsed:.1f}초**")
                                time.sleep(0.1)

                            thread.join()
                            elapsed = time.time() - start_time
                            timer_placeholder.markdown(f"✅ 완료: **{elapsed:.1f}초** 소요")

                            if error_holder[0]:
                                raise error_holder[0]

                            status.update(label=f"✅ 완료 ({elapsed:.1f}초)", state="complete", expanded=False)

                        response = response_holder[0]
                        st.markdown(response.text)
                        st.caption(f"🕐 응답 시간: {elapsed:.1f}초 | 모델: {selected_model_label}")
                        st.session_state.messages.append({"role": "assistant", "content": response.text})

                        # 대화 저장
                        save_qa_to_file(prompt, response.text)
                    except Exception as e:
                        st.error(f"Gemini 오류: {e}")
                else:
                    st.warning("API Key가 설정되어 있지 않아 답변을 생성할 수 없습니다.")
        
        st.rerun()

    st.divider()
    with st.expander("⚙️ Asset Management"):
        st.write("Edit tickers (comma separated)")
        
        new_macro = st.text_area("Global Macro", value=", ".join(macro_ids))
        new_crypto = st.text_area("Crypto", value=", ".join(crypto))
        new_us = st.text_area("US Stocks", value=", ".join(us_stocks))
        new_kr = st.text_area("KR Stocks", value=", ".join(kr_stocks))
        
        st.write("Edit Display Names (JSON format)")
        new_names_json = st.text_area("Display Names Mapping", value=json.dumps(display_names, indent=2, ensure_ascii=False), height=200)
        
        if st.button("Save & Update"):
            try:
                updated_names = json.loads(new_names_json)
                updated_assets = {
                    "macro_ids": [x.strip() for x in new_macro.split(",") if x.strip()],
                    "crypto": [x.strip() for x in new_crypto.split(",") if x.strip()],
                    "us_stocks": [x.strip() for x in new_us.split(",") if x.strip()],
                    "kr_stocks": [x.strip() for x in new_kr.split(",") if x.strip()],
                    "display_names": updated_names
                }
                save_assets(updated_assets)
                st.success("Assets updated!")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON format for Display Names.")

# 2. 메인 분석 영역 (시장별 섹션 분리)
sections = [
    ("🌐 Global Macro Radar Analysis", macro_display, 3),
    ("₿ Crypto Analysis", crypto, 3),
    ("🇺🇸 US Stocks Analysis", us_stocks, 3),
    ("🇰🇷 KR Stocks Analysis", kr_stocks, 3)
]

for section_title, tickers, cols_per_row in sections:
    st.divider()
    st.subheader(section_title)
    
    for i in range(0, len(tickers), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(tickers):
                asset_id = tickers[i + j]
                with row_cols[j]:
                    if asset_id in data_store:
                        d = data_store[asset_id]
                        df = d['df'].tail(60)
                        
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        
                        # 제목 및 요약 정보
                        title = display_names.get(asset_id, d.get('name', asset_id))
                        st.markdown(f'<p class="ticker-title">{title}</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="pattern-label">{d["pattern_label"]}</div>', unsafe_allow_html=True)
                        
                        # RSI 상태 하이라이트
                        rsi_val = d['rsi']
                        rsi_class = "rsi-neutral"
                        if rsi_val >= 70: rsi_class = "rsi-overbought"
                        elif rsi_val <= 30: rsi_class = "rsi-oversold"
                        st.markdown(f"**RSI(14):** <span class='{rsi_class}'>{rsi_val:.1f}</span>", unsafe_allow_html=True)

                        # --- 차트 생성 (Height 상향) ---
                        fig = make_subplots(
                            rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            row_heights=[0.6, 0.2, 0.2]
                        )

                        # 1. 캔들스틱 차트
                        fig.add_trace(go.Candlestick(
                            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                            name='OHLC', increasing_line_color='#00C805', decreasing_line_color='#FF4B4B'
                        ), row=1, col=1)

                        # 2. 하모닉 시각화
                        if d['pattern_points'] is not None:
                            pts = d['pattern_points']
                            x_coords, y_coords = pts.index, pts['Close'].values
                            fig.add_trace(go.Scatter(
                                x=x_coords, y=y_coords, mode='lines+markers+text',
                                text=['X','A','B','C','D'], textposition="top center",
                                line=dict(color='#ECC94B', width=3, dash='dash'),
                                marker=dict(size=10, symbol='diamond', color='#ECC94B'),
                                name='Harmonic'
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=[x_coords[0], x_coords[1], x_coords[2], x_coords[0]],
                                y=[y_coords[0], y_coords[1], y_coords[2], y_coords[0]],
                                fill="toself", fillcolor='rgba(236, 201, 75, 0.1)',
                                line=dict(width=0), showlegend=False
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=[x_coords[2], x_coords[3], x_coords[4], x_coords[2]],
                                y=[y_coords[2], y_coords[3], y_coords[4], y_coords[2]],
                                fill="toself", fillcolor='rgba(236, 201, 75, 0.15)',
                                line=dict(width=0), showlegend=False
                            ), row=1, col=1)

                        # 3. 거래량
                        v_colors = ['#FF4B4B' if c < o else '#00C805' for c, o in zip(df['Close'], df['Open'])]
                        fig.add_trace(go.Bar(
                            x=df.index, y=df['Volume'], marker_color=v_colors, name='Vol'
                        ), row=2, col=1)

                        # 4. RSI 및 기준선
                        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#63B3ED', width=2), name='RSI'), row=3, col=1)
                        fig.add_hline(y=70, line_dash="dot", line_color="#C53030", opacity=0.5, row=3, col=1)
                        fig.add_hline(y=30, line_dash="dot", line_color="#2F855A", opacity=0.5, row=3, col=1)

                        fig.update_layout(
                            height=700, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis_rangeslider_visible=False, showlegend=False,
                            # 모바일 터치 대응을 위한 차트 테두리 추가
                            shapes=[dict(
                                type="rect",
                                xref="paper", yref="paper",
                                x0=0, y0=0, x1=1, y1=1,
                                line=dict(color="#4A5568", width=2)
                            )]
                        )
                        fig.update_yaxes(
                            gridcolor='#2D3748', zeroline=False,
                            showline=True, linewidth=1, linecolor='#4A5568', mirror=True
                        )
                        fig.update_xaxes(
                            gridcolor='#2D3748',
                            showline=True, linewidth=1, linecolor='#4A5568', mirror=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Waiting for {asset_id} data...")

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

# --- 기술적 보조 지표 계산 함수 ---
def compute_technical_indicators(df):
    df = df.copy()
    if len(df) < 40:
        # 데이터가 부족하면 빈 값이나 기본값으로 채움
        df['MA5'] = df['Close']
        df['MA20'] = df['Close']
        df['MA60'] = df['Close']
        df['MA120'] = df['Close']
        df['BB_upper'] = df['Close']
        df['BB_lower'] = df['Close']
        df['MACD'] = 0.0
        df['MACD_signal'] = 0.0
        df['MACD_hist'] = 0.0
        df['OBV'] = 0.0
        return df

    # 이동평균선 (Moving Averages) - 5일(심리), 20일(생명), 60일(수급), 120일(경기)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()

    # 볼린저 밴드 (Bollinger Bands) - 20일선 기준 상하 2표준편차
    std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + 2 * std
    df['BB_lower'] = df['MA20'] - 2 * std

    # MACD - 12일선과 26일선 지수이평선 차이
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # OBV (On-Balance Volume) - 거래량 누적 계산
    obv = [0.0]
    close = df['Close'].values
    volume = df['Volume'].values
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + float(volume[i]))
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - float(volume[i]))
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df

# --- 매매 시그널 추출 및 20일선 눌림목 감지 함수 ---
def extract_signals(df):
    if len(df) < 20:
        return {
            "ma20_trend": "분석 불가능",
            "bb_status": "데이터 부족",
            "macd_status": "데이터 부족",
            "obv_trend": "데이터 부족",
            "candle_pattern": "데이터 부족",
            "pullback_eligible": "아님"
        }
    
    price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    volume = df['Volume'].iloc[-1]
    
    # 1. 20일선(생명선) 추세
    ma20 = df['MA20'].iloc[-1]
    prev_ma20 = df['MA20'].iloc[-2]
    ma20_trend = "상승세 ↗" if ma20 > prev_ma20 else "하락세 ↘"
    
    # 2. 볼린저 밴드 이격도 및 변동성 수축 감지
    bb_upper = df['BB_upper'].iloc[-1]
    bb_lower = df['BB_lower'].iloc[-1]
    bb_width = (bb_upper - bb_lower) / ma20 if ma20 != 0 else 0
    
    # 최근 10일 볼린저 밴드 폭 평균 대비 수축 여부 판단
    prev_bb_widths = ((df['BB_upper'] - df['BB_lower']) / df['MA20']).tail(10)
    avg_bb_width = prev_bb_widths.mean()
    
    bb_status = "중립 ⚪"
    if bb_width < avg_bb_width * 0.9:
        bb_status = "변동성 수축 (Squeeze) ⚡ (에너지 응축 중!)"
    elif price >= bb_upper:
        bb_status = "상한선 돌파 🔴 (과열 가능성)"
    elif price <= bb_lower:
        bb_status = "하한선 이탈 🟢 (과매도 반등 가능)"
        
    # 3. MACD 모멘텀 골든/데드크로스 및 추세 판단
    macd = df['MACD'].iloc[-1]
    macd_sig = df['MACD_signal'].iloc[-1]
    prev_macd = df['MACD'].iloc[-2]
    prev_macd_sig = df['MACD_signal'].iloc[-2]
    
    macd_status = "중립"
    if prev_macd < prev_macd_sig and macd >= macd_sig:
        macd_status = "골든크로스 발생 🔼 (상승 전환!)"
    elif prev_macd > prev_macd_sig and macd <= macd_sig:
        macd_status = "데드크로스 발생 🔽 (하락 주의!)"
    elif macd > 0:
        macd_status = "상승 모멘텀 우세 (0선 위)"
    else:
        macd_status = "하락 모멘텀 우세 (0선 아래)"
        
    # 4. OBV 수급 강도 (매집/분산 감지) - 최근 10영업일 비교
    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] if df['Close'].iloc[-10] != 0 else 0
    obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-10]) / abs(df['OBV'].iloc[-10]) if df['OBV'].iloc[-10] != 0 else 0
    
    obv_trend = "중립 ⚪"
    if abs(price_change) < 0.025 and obv_change > 0.05:
        obv_trend = "세력 매집 포착 💎 (가격 횡보 중 거래량 유입!)"
    elif price_change > 0.05 and obv_change < -0.02:
        obv_trend = "자금 분산 감지 ⚠️ (가격 상승 중 진짜 자금 이탈!)"
    elif obv_change > 0:
        obv_trend = "우상향 ↗ (수급 강세)"
    else:
        obv_trend = "우하향 ↘ (수급 약세)"
        
    # 5. 캔들차트 시장 심리 (도지/망치 지지 확인)
    op = df['Open'].iloc[-1]
    hi = df['High'].iloc[-1]
    lo = df['Low'].iloc[-1]
    cl = df['Close'].iloc[-1]
    body = abs(cl - op)
    rng = hi - lo if hi != lo else 1.0
    
    candle_pattern = "일반형"
    is_doji = body <= rng * 0.1
    is_hammer = (min(op, cl) - lo) >= rng * 0.6 and (hi - max(op, cl)) <= rng * 0.15
    
    if is_doji:
        candle_pattern = "도지(Doji) ✖ (매수/매도 팽팽한 심리적 균형)"
    elif is_hammer:
        candle_pattern = "망치형(Hammer) 🔨 (저가 매수세 유입, 강력 바닥 시그널)"
        
    # 6. 그랜빌의 20일선 눌림목 3대 요건 판정
    # 요건 A: 20일선이 지지를 나타내며 주가가 위에 위치함
    # 요건 B: 20일선 조정 시 거래량이 눈에 띄게 감소함 (최근 5일 평균 거래량의 80% 이하)
    # 요건 C: 도지 또는 망치형 지지 캔들이 출현함
    avg_vol = df['Volume'].tail(5).mean()
    vol_dropped = volume < avg_vol * 0.8
    near_ma20 = ma20 * 0.98 <= cl <= ma20 * 1.02
    
    pullback_eligible = "아님"
    if cl > ma20 and near_ma20 and vol_dropped and (is_doji or is_hammer):
        pullback_eligible = "★ 그랜빌 20일선 눌림목 매수 적격 ★ 🎯 (거래량 급감 + 지지 캔들 확인!)"
        
    return {
        "ma20_trend": ma20_trend,
        "bb_status": bb_status,
        "macd_status": macd_status,
        "obv_trend": obv_trend,
        "candle_pattern": candle_pattern,
        "pullback_eligible": pullback_eligible
    }

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
    X, A, B, C, D = p_vals
    
    # Bullish/Bearish 판별
    is_bullish = X < A and B < A and B > X and C > B and C < A and D < C
    is_bearish = X > A and B > A and B < X and C < B and C > A and D > C

    # 비율 계산
    AB_XA = abs(B-A) / abs(A-X)
    CD_AB = abs(D-C) / abs(B-A) if abs(B-A) != 0 else 0
    
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
            df = compute_technical_indicators(df)
            sig = extract_signals(df)
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
                'pattern_points': pat_points,
                'signals': sig
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
            data = json.load(f)
    except FileNotFoundError:
        # 기본값 반환 및 파일 생성
        data = {
            "macro_ids": [
                "^TNX", "^IRX", "^VIX", "DX-Y.NYB", "GC=F", "CL=F", "SI=F", "^IXIC", "^KS11"
            ],
            "favorites": {
                "관심종목-메인": [
                    "IONQ", "PLTR", "NVDA", "TSLA", "FIG", "GOOGL", "LEU", "COHR", "ASTS", "TEM",
                    "005930.KS", "000660.KS", "017670.KS", "128940.KS", "100790.KQ", "006800.KS", "380550.KQ", "036930.KQ",
                    "BTC-USD", "ETH-USD", "SOL-USD"
                ]
            },
            "display_names": {
                "^TNX": "US 10Y Yield",
                "^IRX": "US 3M T-Bill",
                "^VIX": "VIX 공포지수",
                "DX-Y.NYB": "Dollar Index (DXY)",
                "GC=F": "Gold (금)",
                "CL=F": "Crude Oil WTI (원유)",
                "SI=F": "Silver (은)",
                "^IXIC": "NASDAQ Composite",
                "^KS11": "KOSPI",
                "SPREAD_10Y2Y": "10Y-3M Spread (장단기 금리차)",
                "BTC-USD": "Bitcoin (USD)",
                "ETH-USD": "Ethereum (USD)",
                "SOL-USD": "Solana (USD)",
                "005930.KS": "삼성전자",
                "000660.KS": "SK하이닉스",
                "017670.KS": "SK텔레콤",
                "128940.KS": "한미약품",
                "100790.KQ": "미래에셋벤처투자",
                "006800.KS": "미래에셋증권",
                "380550.KQ": "뉴로핏",
                "036930.KQ": "주성엔지니어링"
            }
        }
        save_assets(data)
        return data

    # 하방 호환성 및 자동 마이그레이션 로직
    migrated = False
    if "favorites" not in data:
        crypto = data.get("crypto", [])
        us_stocks = data.get("us_stocks", [])
        kr_stocks = data.get("kr_stocks", [])
        
        # 합친 관심종목 리스트 생성
        combined_stocks = us_stocks + kr_stocks + crypto
        data["favorites"] = {
            "관심종목-메인": combined_stocks if combined_stocks else []
        }
        
        # 기존 키 제거
        for key in ["crypto", "us_stocks", "kr_stocks"]:
            if key in data:
                del data[key]
        migrated = True
        
    if migrated:
        save_assets(data)
        
    return data

def save_assets(assets):
    with open(ASSETS_FILE, "w", encoding="utf-8") as f:
        json.dump(assets, f, indent=2, ensure_ascii=False)

# 자산 로드
assets_data = load_assets()
macro_ids = assets_data.get("macro_ids", [])
favorites = assets_data.get("favorites", {})
display_names = assets_data.get("display_names", {})

# --- 관심종목 그룹 선택 필터 (Top of Dashboard) ---
existing_groups = list(favorites.keys())
if existing_groups:
    selected_group = st.selectbox(
        "📁 분석할 관심종목 그룹 선택 (Select Group to Analyze)",
        options=["전체 보기"] + existing_groups,
        index=0,
        key="main_group_filter"
    )
else:
    selected_group = "전체 보기"

# 선택된 그룹에 따라 fetch할 액티브 자산 목록만 추출 (드롭다운 변경 시 해당 종목만 로드)
if selected_group == "전체 보기":
    active_tickers = list(macro_ids)
    for group_name, tickers in favorites.items():
        active_tickers.extend(tickers)
else:
    active_tickers = list(macro_ids) + favorites.get(selected_group, [])

# 중복 제거 및 순서 보존
active_tickers = list(dict.fromkeys(active_tickers))

# 데이터 로드 (선택한 그룹의 종목들만 실시간 fetch 실행)
data_store = fetch_all_assets(active_tickers)

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
        "⚡ Gemini 3.5 Flash (GA)": "gemini-3.5-flash",
        "💨 Gemini 3.1 Flash-Lite (GA)": "gemini-3.1-flash-lite",
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
                        # 현재 데이터 컨텍스트 생성 (고급 보조 지표 및 눌림목 감지 상태 포함)
                        context = "현재 분석 중인 상세 종목 기술 지표 데이터:\n"
                        if data_store:
                            for tid, stats in data_store.items():
                                stock_name = display_names.get(tid) or stats.get('name') or tid
                                sig = stats.get('signals', {})
                                context += (
                                    f"- {stock_name} ({tid}): 현재가 {stats['price']:.2f}, "
                                    f"RSI(14) {stats['rsi']:.1f}, "
                                    f"20일생명선 추세 [{sig.get('ma20_trend')}], "
                                    f"볼린저밴드 [{sig.get('bb_status')}], "
                                    f"MACD [{sig.get('macd_status')}], "
                                    f"OBV자금수급 [{sig.get('obv_trend')}], "
                                    f"캔들패턴 [{sig.get('candle_pattern')}], "
                                    f"그랜빌눌림목판정 [{sig.get('pullback_eligible')}], "
                                    f"하모닉패턴 [{stats['pattern_label']}]\n"
                                )
                        else:
                            context += "데이터 로딩 중...\n"

                        system_prompt = f"""너는 20년 경력의 시니어 퀀트 애널리스트이자 기술적 분석 대가인 'Quantum Sidekick'이야.
다음 데이터를 참고하여 사용자의 투자 질의에 전문적이고 친절하게 답변해줘.

너는 항상 전통적인 차트 원칙과 퀀트 분석 기준인 **[주식 차트 분석 및 실전 매매 가이드]**의 다음 5대 규칙에 기반하여 분석해야 한다:

1. **캔들 차트의 본질 & 사카타 5법**: 캔들은 대중 심리의 결과물이다. '도지(Doji)'는 팽팽한 힘의 균형/방향성 탐색을 의미하고, '망치형(Hammer)'은 강력한 하단 지지 신호다. 삼산(헤드앤숄더)은 강력한 천정 및 하락 전환, 삼천(역헤드앤숄더)은 강력한 바닥 및 상승 전환 신호다.
2. **그랜빌 8법칙 & 20일 생명선 눌림목**: 20일 이동평균선(생명선)이 우상향하고 주가가 그 위에 있을 때, 일시 조정으로 20일선 근처까지 내려왔을 때 거래량이 눈에 띄게 급감(Vol Dropped)하고 도지 또는 망치형 캔들로 지지가 확인되면 '그랜빌 20일선 눌림목 매수 적격(Pullback Eligible)' 신호가 된다.
3. **OBV (On-Balance Volume) 거래량 수급**: 주가가 횡보하는데 OBV선이 꾸준히 상승하면 세력의 '매집' 신호로 급등 가능성이 크고, 주가가 고점을 높이는데 OBV선이 낮아지거나 꺾이면 개인들만 매수하는 '분산(이탈)' 신호로 하락 전환 임박 경고다.
4. **볼린저 밴드 변동성**: 밴드가 좁혀지는 수축(Squeeze)은 강력한 에너지 응축 구간으로 조만간 큰 추세 돌파가 나옴을 뜻한다. 상한선 돌파는 단기 과열, 하한선 돌파는 단기 과매도 상태다.
5. **MACD & RSI 다차원 결합**: 추세(이평선) x 모멘텀(MACD 0선 돌파 및 골든/데드크로스) x 변동성(RSI 30이하 과매도, 70이상 과매수)을 결합하여 분석하라. 특히 주가는 고점을 높이나 보조지표 고점은 낮아지는 일반 다이버전스는 강력한 반전 신호다.

[실시간 데이터 컨텍스트]
{context}

사용자 질문에 답변할 때, 위 데이터에 들어있는 RSI, 20일선 추세, 볼린저 밴드 상태, MACD 모멘텀, OBV 수급 강도, 눌림목 판정 정보를 적극 인용하여 애널리스트처럼 설득력 있게 한글로 설명해줘.
"""

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
    with st.expander("⚙️ Asset & Group Management"):
        tab1, tab2, tab3 = st.tabs(["📝 관심종목 등록", "🗑️ 삭제/그룹 관리", "🌐 매크로 & 상세"])
        
        with tab1:
            st.markdown("#### ➕ 새로운 관심종목 등록")
            
            # 그룹 목록 추출
            existing_groups = list(favorites.keys())
            if not existing_groups:
                existing_groups = ["관심종목-메인"]
                
            group_options = existing_groups + ["+ 새 그룹 추가..."]
            
            selected_group_opt = st.selectbox(
                "📁 등록할 그룹 선택",
                options=group_options,
                index=0,
                key="reg_group_select"
            )
            
            new_group_name = ""
            if selected_group_opt == "+ 새 그룹 추가...":
                new_group_name = st.text_input(
                    "✏️ 새 그룹 이름 입력",
                    placeholder="예: 미국 성장주, 반도체",
                    key="reg_new_group_input"
                ).strip()
            
            reg_ticker = st.text_input(
                "🔤 종목 코드 (Ticker Symbol)",
                placeholder="예: AAPL, 005930.KS, TSLA",
                key="reg_ticker_input"
            ).strip().upper()
            
            reg_name = st.text_input(
                "🏷️ 표시 이름 (Display Name - 선택사항)",
                placeholder="예: 애플, 삼성전자",
                key="reg_name_input"
            ).strip()
            
            if st.button("🚀 종목 등록", use_container_width=True):
                # 유효성 검사
                target_group = new_group_name if selected_group_opt == "+ 새 그룹 추가..." else selected_group_opt
                
                if not target_group:
                    st.error("⚠️ 그룹 이름을 입력하거나 선택해주세요.")
                elif not reg_ticker:
                    st.error("⚠️ 종목 코드를 입력해주세요.")
                else:
                    # 데이터 업데이트
                    if target_group not in favorites:
                        favorites[target_group] = []
                    
                    if reg_ticker not in favorites[target_group]:
                        favorites[target_group].append(reg_ticker)
                        st.success(f"✅ {reg_ticker}가 '{target_group}' 그룹에 추가되었습니다!")
                    else:
                        st.info(f"ℹ️ {reg_ticker}는 이미 '{target_group}' 그룹에 등록되어 있습니다.")
                    
                    # 표시 이름이 설정되어 있다면 추가
                    if reg_name:
                        display_names[reg_ticker] = reg_name
                    
                    # 저장 및 새로고침
                    assets_data["favorites"] = favorites
                    assets_data["display_names"] = display_names
                    save_assets(assets_data)
                    st.rerun()
                    
        with tab2:
            st.markdown("#### 🗑️ 관심종목 및 그룹 관리")
            if favorites:
                del_group = st.selectbox(
                    "📁 관리할 그룹 선택",
                    options=list(favorites.keys()),
                    key="del_group_select"
                )
                
                group_tickers = favorites.get(del_group, [])
                
                if group_tickers:
                    del_ticker = st.selectbox(
                        "❌ 삭제할 종목 선택",
                        options=group_tickers,
                        format_func=lambda x: f"{display_names.get(x, x)} ({x})",
                        key="del_ticker_select"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ 종목 삭제", use_container_width=True, type="secondary"):
                            favorites[del_group].remove(del_ticker)
                            assets_data["favorites"] = favorites
                            save_assets(assets_data)
                            st.success(f"✅ {del_ticker} 삭제 완료!")
                            st.rerun()
                    with col2:
                        if st.button("🔥 그룹 전체 삭제", use_container_width=True, type="primary"):
                            del favorites[del_group]
                            assets_data["favorites"] = favorites
                            save_assets(assets_data)
                            st.success(f"✅ '{del_group}' 그룹이 통째로 삭제되었습니다!")
                            st.rerun()
                else:
                    st.info("이 그룹에는 등록된 종목이 없습니다.")
                    if st.button("🔥 빈 그룹 삭제", use_container_width=True, type="primary"):
                        del favorites[del_group]
                        assets_data["favorites"] = favorites
                        save_assets(assets_data)
                        st.success(f"✅ '{del_group}' 그룹이 삭제되었습니다!")
                        st.rerun()
            else:
                st.warning("⚠️ 등록된 관심종목 그룹이 없습니다.")
                
        with tab3:
            st.markdown("#### 🌐 글로벌 매크로 인덱스 편집")
            new_macro = st.text_area(
                "쉼표(,)로 구분된 Tickers",
                value=", ".join(macro_ids),
                help="예: ^TNX, ^VIX, GC=F",
                key="macro_text_area"
            )
            
            st.markdown("#### 📋 표시 이름 일괄 수정 (JSON)")
            new_names_json = st.text_area(
                "Display Names Mapping",
                value=json.dumps(display_names, indent=2, ensure_ascii=False),
                height=200,
                key="display_names_json_area"
            )
            
            if st.button("💾 매크로 및 이름 저장", use_container_width=True):
                try:
                    updated_names = json.loads(new_names_json)
                    updated_macro = [x.strip() for x in new_macro.split(",") if x.strip()]
                    
                    assets_data["macro_ids"] = updated_macro
                    assets_data["display_names"] = updated_names
                    save_assets(assets_data)
                    st.success("✅ 설정이 저장되었습니다!")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("⚠️ 표시 이름 매핑 JSON 형식이 올바르지 않습니다.")

# 2. 메인 분석 영역 (관심종목 그룹 필터 및 시장별 섹션 분리)
sections = []

# "전체 보기"일 때만 글로벌 매크로 분석 섹션을 포함
if selected_group == "전체 보기":
    sections.append(("🌐 Global Macro Radar Analysis", macro_display, 3))

# 선택된 그룹만 sections에 동적 추가
for group_name, tickers in favorites.items():
    if tickers:  # 비어있는 그룹은 표시하지 않음
        if selected_group == "전체 보기" or selected_group == group_name:
            sections.append((f"⭐ {group_name}", tickers, 3))

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
                        
                        # RSI 및 기술적 보조 지표 상태 하이라이트
                        rsi_val = d['rsi']
                        rsi_class = "rsi-neutral"
                        if rsi_val >= 70: rsi_class = "rsi-overbought"
                        elif rsi_val <= 30: rsi_class = "rsi-oversold"
                        
                        sig = d.get('signals', {})
                        st.markdown(f"""
                        *   **RSI (14)**: <span class='{rsi_class}'>{rsi_val:.1f}</span>
                        *   **20일 생명선**: {sig.get('ma20_trend', 'N/A')}
                        *   **볼린저 밴드**: {sig.get('bb_status', 'N/A')}
                        *   **MACD 모멘텀**: {sig.get('macd_status', 'N/A')}
                        *   **OBV 자금수급**: {sig.get('obv_trend', 'N/A')}
                        *   **심리 캔들패턴**: {sig.get('candle_pattern', 'N/A')}
                        """, unsafe_allow_html=True)
                        
                        if sig.get('pullback_eligible') != "아님":
                            st.markdown(f"🔥 **그랜빌 눌림목**: <span style='color: #ECC94B; font-weight: bold;'>{sig.get('pullback_eligible')}</span>", unsafe_allow_html=True)

                        # --- 차트 생성 (Height 상향) ---
                        fig = make_subplots(
                            rows=3, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            row_heights=[0.6, 0.2, 0.2]
                        )

                        # 1. 볼린저 밴드 영역 채우기 오버레이 (가장 뒷배경에 그리기 위해 맨 처음 추가)
                        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df['BB_upper'], line=dict(color='rgba(113, 128, 150, 0.25)', width=1, dash='dot'),
                                name='BB Upper', showlegend=False
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df['BB_lower'], line=dict(color='rgba(113, 128, 150, 0.25)', width=1, dash='dot'),
                                name='BB Lower', fill='tonexty', fillcolor='rgba(113, 128, 150, 0.04)',
                                showlegend=False
                            ), row=1, col=1)

                        # 2. 이동평균선 오버레이 (5일선: 심리선, 20일선: 생명선, 60일선: 수급선)
                        if 'MA5' in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df['MA5'], line=dict(color='#4299E1', width=1.2, dash='dash'),
                                name='5일선(심리선)'
                            ), row=1, col=1)
                        if 'MA20' in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df['MA20'], line=dict(color='#ED8936', width=2.2),
                                name='20일선(생명선)'
                            ), row=1, col=1)
                        if 'MA60' in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df['MA60'], line=dict(color='#38A169', width=1.2),
                                name='60일선(수급선)'
                            ), row=1, col=1)

                        # 3. 캔들스틱 차트
                        fig.add_trace(go.Candlestick(
                            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                            name='OHLC', increasing_line_color='#00C805', decreasing_line_color='#FF4B4B'
                        ), row=1, col=1)

                        # 4. 하모닉 시각화
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

                        # 5. 거래량
                        v_colors = ['#FF4B4B' if c < o else '#00C805' for c, o in zip(df['Close'], df['Open'])]
                        fig.add_trace(go.Bar(
                            x=df.index, y=df['Volume'], marker_color=v_colors, name='Vol'
                        ), row=2, col=1)

                        # 6. RSI 및 기준선
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
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"chart_{section_title}_{asset_id}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Waiting for {asset_id} data...")

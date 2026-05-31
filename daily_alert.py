"""
daily_alert.py
--------------
매일 아침 8시에 자동 실행되는 주식 분석 스크립트.
Streamlit 없이 독립 실행 가능. GitHub Actions로 스케줄링.

실행: python daily_alert.py
"""

import os
import sys
import json
import requests
from datetime import datetime
import yfinance as yf
import pandas as pd
# pandas_ta는 Python 3.11 Linux 환경에서 미지원 → 직접 RSI 계산
import numpy as np
from scipy.signal import argrelextrema
from google import genai
from dotenv import load_dotenv

# ── 환경 변수 로드 ──────────────────────────────────────────────
# GitHub Actions: .env 파일 없음 → os.environ에서 직접 읽음
load_dotenv()

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
ASSETS_FILE       = os.path.join(os.path.dirname(__file__), "assets.json")

# ── API 키 검증 (없으면 즉시 종료) ──────────────────────────────
if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY가 설정되지 않았습니다. GitHub Secrets를 확인하세요.")
    sys.exit(1)
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("[ERROR] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
    sys.exit(1)

# ── Gemini 초기화 (키 검증 후 실행) ─────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)
print(f"[OK] Gemini 초기화 완료: {GEMINI_MODEL}")


# ── 하모닉 패턴 감지 (app.py와 동일 로직) ────────────────────────
def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """pandas만으로 RSI 계산 (pandas_ta 대체)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── 기술적 보조 지표 계산 함수 ───────────────────────────────────
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 40:
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

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()

    std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + 2 * std
    df['BB_lower'] = df['MA20'] - 2 * std

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

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


# ── 매매 시그널 추출 및 20일선 눌림목 감지 함수 ─────────────────
def extract_signals(df: pd.DataFrame) -> dict:
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
    volume = df['Volume'].iloc[-1]
    
    ma20 = df['MA20'].iloc[-1]
    prev_ma20 = df['MA20'].iloc[-2]
    ma20_trend = "상승세 ↗" if ma20 > prev_ma20 else "하락세 ↘"
    
    bb_upper = df['BB_upper'].iloc[-1]
    bb_lower = df['BB_lower'].iloc[-1]
    bb_width = (bb_upper - bb_lower) / ma20 if ma20 != 0 else 0
    
    prev_bb_widths = ((df['BB_upper'] - df['BB_lower']) / df['MA20']).tail(10)
    avg_bb_width = prev_bb_widths.mean()
    
    bb_status = "중립 ⚪"
    if bb_width < avg_bb_width * 0.9:
        bb_status = "변동성 수축 (Squeeze) ⚡ (에너지 응축!)"
    elif price >= bb_upper:
        bb_status = "상한선 돌파 🔴 (과열 가능성)"
    elif price <= bb_lower:
        bb_status = "하한선 이탈 🟢 (과매도 반등 가능)"
        
    macd = df['MACD'].iloc[-1]
    macd_sig = df['MACD_signal'].iloc[-1]
    prev_macd = df['MACD'].iloc[-2]
    prev_macd_sig = df['MACD_signal'].iloc[-2]
    
    macd_status = "중립"
    if prev_macd < prev_macd_sig and macd >= macd_sig:
        macd_status = "골든크로스 발생 🔼"
    elif prev_macd > prev_macd_sig and macd <= macd_sig:
        macd_status = "데드크로스 발생 🔽"
    elif macd > 0:
        macd_status = "상승 모멘텀 우세 (0선 위)"
    else:
        macd_status = "하락 모멘텀 우세 (0선 아래)"
        
    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] if df['Close'].iloc[-10] != 0 else 0
    obv_change = (df['OBV'].iloc[-1] - df['OBV'].iloc[-10]) / abs(df['OBV'].iloc[-10]) if df['OBV'].iloc[-10] != 0 else 0
    
    obv_trend = "중립 ⚪"
    if abs(price_change) < 0.025 and obv_change > 0.05:
        obv_trend = "세력 매집 포착 💎 (횡보 중 OBV 유입!)"
    elif price_change > 0.05 and obv_change < -0.02:
        obv_trend = "자금 분산 감지 ⚠️ (상승 중 진짜 자금 이탈!)"
    elif obv_change > 0:
        obv_trend = "우상향 ↗ (수급 강세)"
    else:
        obv_trend = "우하향 ↘ (수급 약세)"
        
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
        candle_pattern = "도지(Doji) ✖"
    elif is_hammer:
        candle_pattern = "망치형(Hammer) 🔨"
        
    avg_vol = df['Volume'].tail(5).mean()
    vol_dropped = volume < avg_vol * 0.8
    near_ma20 = ma20 * 0.98 <= cl <= ma20 * 1.02
    
    pullback_eligible = "아님"
    if cl > ma20 and near_ma20 and vol_dropped and (is_doji or is_hammer):
        pullback_eligible = "★ 그랜빌 20일선 눌림목 매수 적격 ★ 🎯 (거래량 급감 + 지지 캔들!)"
        
    return {
        "ma20_trend": ma20_trend,
        "bb_status": bb_status,
        "macd_status": macd_status,
        "obv_trend": obv_trend,
        "candle_pattern": candle_pattern,
        "pullback_eligible": pullback_eligible
    }


def detect_patterns(df):
    if len(df) < 40:
        return "Insufficient Data", None

    n = 5
    df = df.copy()
    df['min'] = df['Close'].iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]
    df['max'] = df['Close'].iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]

    points = df.dropna(subset=['min', 'max'], how='all')
    if len(points) < 5:
        return "No Pattern", None

    last_5 = points.tail(5)
    p_vals = last_5['Close'].values
    X, A, B, C, D = p_vals

    is_bullish = X < A and B < A and B > X and C > B and C < A and D < C
    is_bearish = X > A and B > A and B < X and C < B and C > A and D > C

    AB_XA = abs(B - A) / abs(A - X) if abs(A - X) != 0 else 0
    CD_AB = abs(D - C) / abs(B - A) if abs(B - A) != 0 else 0

    if 0.58 < AB_XA < 0.65:
        pattern_type = "Gartley"
    elif 0.38 < AB_XA < 0.52:
        pattern_type = "Bat"
    elif 0.75 < AB_XA < 0.82:
        pattern_type = "Butterfly"
    else:
        pattern_type = "Complex Structure"

    direction = "Bullish 🔼" if is_bullish else ("Bearish 🔽" if is_bearish else "Neutral")
    abcd_status = "AB=CD OK" if 0.88 < CD_AB < 1.12 else f"AB=CD ratio {CD_AB:.2f}"

    return f"{pattern_type} | {direction} | {abcd_status}", last_5


# ── 주식 데이터 수집 ─────────────────────────────────────────────
def fetch_stock_data(tickers: list, display_names: dict = {}) -> dict:
    data = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="6mo")
            if df.empty:
                continue

            df['RSI'] = calc_rsi(df['Close'], length=14)
            df = compute_technical_indicators(df)
            sig = extract_signals(df)
            pat_label, _ = detect_patterns(df.copy())

            price   = df['Close'].iloc[-1]
            prev    = df['Close'].iloc[-2]
            change  = (price - prev) / prev * 100
            rsi     = df['RSI'].iloc[-1]

            # RSI 신호
            if rsi <= 30:
                rsi_signal = "🟢 과매도(매수 고려)"
            elif rsi >= 70:
                rsi_signal = "🔴 과열(매도 고려)"
            else:
                rsi_signal = "⚪ 중립"

            # 이름 및 링크 설정
            name = display_names.get(t, t)
            if t.endswith((".KS", ".KQ")):
                code = t.split(".")[0]
                url = f"https://finance.naver.com/item/main.naver?code={code}"
            else:
                url = f"https://finance.yahoo.com/quote/{t}"

            data[t] = {
                "name":       name,
                "url":        url,
                "price":      price,
                "change_pct": change,
                "rsi":        rsi,
                "rsi_signal": rsi_signal,
                "pattern":    pat_label,
                "ma20_trend": sig["ma20_trend"],
                "bb_status":  sig["bb_status"],
                "macd_status": sig["macd_status"],
                "obv_trend":  sig["obv_trend"],
                "candle_pattern": sig["candle_pattern"],
                "pullback_eligible": sig["pullback_eligible"]
            }
            print(f"  [OK] {name}: {price:.2f} ({change:+.1f}%) RSI={rsi:.1f}")
        except Exception as e:
            print(f"  [ERROR] {t} 오류: {e}")
    return data


# ── Gemini 매수/매도 분석 ─────────────────────────────────────────
def analyze_with_gemini(data: dict) -> str:
    today = datetime.now().strftime("%Y년 %m월 %d일")

    # 데이터 요약 텍스트 생성 (고급 보조 지표 및 눌림목 감지 상태 포함)
    summary_lines = []
    for ticker, d in data.items():
        summary_lines.append(
            f"- {d['name']} ({d['url']}): 현재가 {d['price']:.2f}, "
            f"등락 {d['change_pct']:+.1f}%, "
            f"RSI(14) {d['rsi']:.1f} ({d['rsi_signal']}), "
            f"20일생명선 추세 [{d['ma20_trend']}], "
            f"볼린저밴드 [{d['bb_status']}], "
            f"MACD [{d['macd_status']}], "
            f"OBV자금수급 [{d['obv_trend']}], "
            f"캔들패턴 [{d['candle_pattern']}], "
            f"그랜빌눌림목판정 [{d['pullback_eligible']}], "
            f"하모닉패턴 [{d['pattern']}]"
        )
    data_text = "\n".join(summary_lines)

    prompt = f"""너는 20년 경력의 시니어 퀀트 애널리스트야.
오늘({today}) 아침 기준 아래 종목 데이터를 분석해서 매수/매도/관망 판단을 내려줘.

이 분석은 전통적인 차트 원칙과 퀀트 분석 기준인 **[주식 차트 분석 및 실전 매매 가이드]**의 핵심 사상에 기반한다.
주식을 분석할 때 아래 원칙들을 유기적으로 결합하여 최종 판정을 내려라:

[분석 및 매매 원칙]
1. **사카타 5법 & 캔들 본질**: 캔들은 시장 심리의 기록이다. 도지(Doji)는 방향성 모색/힘의 균형을 나타내고, 아래꼬리가 긴 망치형(Hammer)은 저가 매수세의 바닥 지지 신호다. 삼산(헤드앤숄더)은 강력한 천정 및 하락 전환 신호이며, 삼천(역헤드앤숄더)은 강력한 바닥 및 상승 전환 신호다.
2. **그랜빌 8법칙 & 20일선 눌림목**: 상승 추세(20일선 우상향)인 종목이 일시 조정으로 20일선 근처에 도달하고, 거래량이 눈에 띄게 급감(Vol Dropped)하며, 도지 또는 망치형 캔들로 지지가 확인될 때가 가장 확률 높은 "눌림목 매수 적격(Pullback Eligible)" 적기다.
3. **OBV 매집/분산 감지**: 가격이 횡보하는데 OBV가 상승하면 세력의 '매집' 신호(매수 극대화), 가격은 계속 상승하는데 OBV가 하락하거나 둔화하면 진짜 상승 에너지가 빠지는 '분산(이탈)' 신호(매도/관망)다.
4. **볼린저 밴드 변동성**: 밴드가 매우 좁혀지는 수축(Squeeze)은 강력한 에너지 응축 구간으로 조만간 큰 변동성 돌파가 나옴을 뜻한다. 상한선 돌파는 단기 과열(매도 고려), 하한선 이탈은 단기 과매도(반등 매수 고려) 상태다.
5. **MACD & RSI 다차원 결합**: 추세(이평선) x 모멘텀(MACD 0선 돌파 및 골든/데드크로스) x 과열도(RSI 30이하 과매도, 70이상 과매수)를 결합하여 분석하라. 주가 고점은 오르는데 보조지표 고점이 낮아지는 일반 다이버전스는 강력한 추세 하락 반전 신호다.

[종목 데이터]
{data_text}

[출력 형식 - 반드시 아래 형식으로만 답변]
각 종목마다 한 줄:
🟢 매수 [주식명](링크): [이유 한 줄 - 사카타 5법/그랜빌 눌림목/OBV 매집/RSI 과매도 등 위의 분석 원칙 근거들을 직접 인용하여 설득력 있게 설명]
🔴 매도 [주식명](링크): [이유 한 줄 - 과열/자금 분산/데드크로스/이탈 등 기술적 근거를 설명]
⚪ 관망 [주식명](링크): [이유 한 줄 - 횡보/방향성 탐색/삼법 횡보 구간 등 근거를 설명]

* 주식명은 데이터에 제공된 이름을 사용하고, 링크도 그대로 포함해줘.
* 판단 근거 기술 시 "20일선 지지 및 거래량 급감 확인", "OBV 매집 포착", "RSI 과매도 구간 진입", "하모닉 패턴 감지" 등 배운 차트 분석 용어를 직접 언급하여 신뢰성 있게 적어줘.

마지막에 오늘의 시장 총평과 주요 투자 포인트(10Y-3M 장단기 금리차 스프레드 상황 포함)를 3줄로 작성해줘.
"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# ── Telegram 발송 ────────────────────────────────────────────────
def send_telegram(text: str) -> bool:
    """Telegram Bot API로 메시지 발송. 4096자 초과 시 분할 전송."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    max_len = 4000

    # 긴 메시지 분할
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    success = True

    for chunk in chunks:
        payload = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       chunk,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            print(f"  [ERROR] Telegram 오류: {resp.status_code} {resp.text}")
            success = False
        else:
            print("  [OK] Telegram 발송 성공")

    return success


# ── 메인 실행 ────────────────────────────────────────────────────
def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*50}")
    print(f"[INFO] Daily Stock Alert 시작: {now}")
    print(f"{'='*50}")

    # 1. 종목 목록 로드
    try:
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            assets = json.load(f)
    except FileNotFoundError:
        print("assets.json 없음 → 기본 종목 사용")
        assets = {
            "macro_ids": ["^TNX", "^IRX", "^VIX", "DX-Y.NYB", "GC=F", "CL=F", "SI=F", "^IXIC", "^KS11"],
            "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "us_stocks": ["IONQ", "PLTR", "NVDA", "TSLA"],
            "kr_stocks": ["017670.KS", "128940.KS"],
        }

    display_names = assets.get("display_names", {})

    all_tickers = list(assets.get("macro_ids", []))
    if "favorites" in assets:
        for group_name, tickers in assets["favorites"].items():
            for t in tickers:
                if t not in all_tickers:
                    all_tickers.append(t)
    else:
        all_tickers.extend(assets.get("crypto", []))
        all_tickers.extend(assets.get("us_stocks", []))
        all_tickers.extend(assets.get("kr_stocks", []))
    print(f"\n[1/3] 종목 데이터 수집 중... ({len(all_tickers)}개)")
    data = fetch_stock_data(all_tickers, display_names)

    # 10Y-3M 장단기 금리차 스프레드 계산
    if "^TNX" in data and "^IRX" in data:
        try:
            spread_val = data["^TNX"]["price"] - data["^IRX"]["price"]
            spread_chg = data["^TNX"]["change_pct"] - data["^IRX"]["change_pct"]
            rsi_val    = (data["^TNX"]["rsi"] + data["^IRX"]["rsi"]) / 2
            data["SPREAD_10Y2Y"] = {
                "name":       display_names.get("SPREAD_10Y2Y", "10Y-3M Spread"),
                "url":        "https://fred.stlouisfed.org/series/T10Y3M",
                "price":      spread_val,
                "change_pct": spread_chg,
                "rsi":        rsi_val,
                "rsi_signal": "⚪ 중립",
                "pattern":    f"10Y({data['^TNX']['price']:.2f}%) - 3M({data['^IRX']['price']:.2f}%)",
                "ma20_trend": "계산 제외",
                "bb_status":  "계산 제외",
                "macd_status": "계산 제외",
                "obv_trend":  "계산 제외",
                "candle_pattern": "해당 없음",
                "pullback_eligible": "아님"
            }
            print(f"  [OK] SPREAD_10Y2Y: {spread_val:+.2f}%p")
        except Exception as e:
            print(f"  [ERROR] 스프레드 계산 오류: {e}")

    if not data:
        print("데이터 수집 실패. 종료.")
        return

    # 2. Gemini 분석
    print("\n[2/3] Gemini AI 매수/매도 분석 중...")
    analysis = analyze_with_gemini(data)
    print("  [OK] 분석 완료")

    # 3. Telegram 발송
    header = f"*📊 QuantumBrief 매일 아침 분석 리포트*\n_{now} 기준_\n\n"
    message = header + analysis
    print("\n[3/3] Telegram 발송 중...")
    send_telegram(message)

    print(f"\n{'='*50}")
    print("[OK] Daily Alert 완료!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

"""
daily_alert.py
--------------
매일 아침 8시에 자동 실행되는 주식 분석 스크립트.
Streamlit 없이 독립 실행 가능. GitHub Actions로 스케줄링.

실행: python daily_alert.py
"""

import os
import json
import requests
from datetime import datetime
import yfinance as yf
import pandas as pd
# pandas_ta는 Python 3.11 Linux 환경에서 미지원 → 직접 RSI 계산
import numpy as np
from scipy.signal import argrelextrema
import google.generativeai as genai
from dotenv import load_dotenv

# ── 환경 변수 로드 ──────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
TELEGRAM_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID")
ASSETS_FILE       = os.path.join(os.path.dirname(__file__), "assets.json")

# ── Gemini 초기화 ────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


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
def fetch_stock_data(tickers: list) -> dict:
    data = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="6mo")
            if df.empty:
                continue

            df['RSI'] = calc_rsi(df['Close'], length=14)
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

            data[t] = {
                "price":      price,
                "change_pct": change,
                "rsi":        rsi,
                "rsi_signal": rsi_signal,
                "pattern":    pat_label,
            }
            print(f"  ✓ {t}: {price:.2f} ({change:+.1f}%) RSI={rsi:.1f}")
        except Exception as e:
            print(f"  ✗ {t} 오류: {e}")
    return data


# ── Gemini 매수/매도 분석 ─────────────────────────────────────────
def analyze_with_gemini(data: dict) -> str:
    today = datetime.now().strftime("%Y년 %m월 %d일")

    # 데이터 요약 텍스트 생성
    summary_lines = []
    for ticker, d in data.items():
        summary_lines.append(
            f"- {ticker}: 현재가 {d['price']:.2f}, "
            f"등락 {d['change_pct']:+.1f}%, "
            f"RSI {d['rsi']:.1f} ({d['rsi_signal']}), "
            f"패턴 [{d['pattern']}]"
        )
    data_text = "\n".join(summary_lines)

    prompt = f"""너는 20년 경력의 시니어 퀀트 애널리스트야.
오늘({today}) 아침 기준 아래 종목 데이터를 분석해서 매수/매도/관망 판단을 내려줘.

[종목 데이터]
{data_text}

[판단 기준]
- RSI 30 이하: 과매도 → 매수 고려
- RSI 70 이상: 과열 → 매도/관망 고려
- Bullish 패턴: 매수 신호 강화
- Bearish 패턴: 매도 신호 강화

[출력 형식 - 반드시 아래 형식으로만 답변]
각 종목마다 한 줄:
🟢 매수 [티커]: 이유 한 줄
🔴 매도 [티커]: 이유 한 줄
⚪ 관망 [티커]: 이유 한 줄

마지막에 오늘의 시장 총평을 2~3줄로 작성해줘.
"""

    response = model.generate_content(prompt)
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
            print(f"  ✗ Telegram 오류: {resp.status_code} {resp.text}")
            success = False
        else:
            print("  ✓ Telegram 발송 성공")

    return success


# ── 메인 실행 ────────────────────────────────────────────────────
def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*50}")
    print(f"📊 Daily Stock Alert 시작: {now}")
    print(f"{'='*50}")

    # 1. 종목 목록 로드
    try:
        with open(ASSETS_FILE, "r", encoding="utf-8") as f:
            assets = json.load(f)
    except FileNotFoundError:
        print("assets.json 없음 → 기본 종목 사용")
        assets = {
            "macro_ids": ["^TNX", "BTC-USD"],
            "us_stocks": ["IONQ", "PLTR", "NVDA", "TSLA"],
            "kr_stocks": ["017670.KS", "128940.KS"],
        }

    all_tickers = (
        assets.get("macro_ids", []) +
        assets.get("us_stocks", []) +
        assets.get("kr_stocks", [])
    )
    print(f"\n[1/3] 종목 데이터 수집 중... ({len(all_tickers)}개)")
    data = fetch_stock_data(all_tickers)

    if not data:
        print("데이터 수집 실패. 종료.")
        return

    # 2. Gemini 분석
    print("\n[2/3] Gemini AI 매수/매도 분석 중...")
    analysis = analyze_with_gemini(data)
    print("  ✓ 분석 완료")

    # 3. Telegram 발송
    header = f"*📊 QuantumBrief 매일 아침 분석 리포트*\n_{now} 기준_\n\n"
    message = header + analysis
    print("\n[3/3] Telegram 발송 중...")
    send_telegram(message)

    print(f"\n{'='*50}")
    print("✅ Daily Alert 완료!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

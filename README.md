# 💎 QuantumBrief Pro

**QuantumBrief Pro**는 하모닉 패턴(Harmonic Patterns) 기술적 분석과 Google Gemini AI를 결합한 지능형 주식 분석 대시보드입니다. 실시간 시장 데이터 분석부터 AI 애널리스트와의 상담, 그리고 상담 결과의 Notion 자동 저장까지 지원합니다.

## ✨ 주요 기능

- **실시간 기술적 분석**: 하모닉 패턴(Gartley, Bat, Butterfly 등) 자동 탐지 및 RSI 지표 제공
- **AI 퀀트 애널리스트**: 대시보드 데이터를 기반으로 종목에 대해 질문하고 전문적인 답변 획득 (Gemini 1.5 Flash 모델 활용)
- **영구 데이터 저장**: AI와의 대화 내용을 로컬 마크다운 파일(`java/answer/`) 및 **Notion 데이터베이스**에 동시 자동 저장
- **글로벌 자산 모니터링**: 국장(KOSPI/KOSDAQ), 미장(NASDAQ/NYSE), 비트코인, 국채 금리 통합 관제

## 🛠 설치 및 로컬 실행 방법

1. **저장소 클론**

   ```bash
   git clone https://github.com/YOUR_USERNAME/Briefing.git
   cd Briefing
   ```

2. **가상환경 설정 및 패키지 설치**

   ```bash
   python -m venv venv
   source venv/bin/scripts/activate  # Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **환경 변수 설정**
   프로젝트 루트 폴더에 `.env` 파일을 생성하고 아래 형식을 입력합니다. (이 파일은 절대 공개되지 않도록 주의하세요!)

   ```text
   GEMINI_API_KEY=your_gemini_api_key
   NOTION_TOKEN=your_notion_integration_token
   NOTION_DATABASE_ID=your_notion_database_id
   ```

4. **앱 실행**
   ```bash
   streamlit run app.py
   ```

## 🚀 배포 및 자동 저장 설정 (Streamlit Cloud & Notion)

### 1. Notion 연동 설정

- [Notion Developers](https://www.notion.so/my-integrations)에서 새 통합(Integration)을 생성합니다.
- 분석 결과를 저장할 Notion 데이터베이스를 만들고, 우측 상단 `...` -> `연결 추가`에서 생성한 통합을 허용합니다.
- 데이터베이스 첫 번째 컬럼(Title) 이름을 **"주식 분석"**으로, 두 번째 컬럼을 **"날짜"**(Date 유형)로 설정합니다.

### 2. Streamlit Cloud 배포

- GitHub 저장소를 연결하여 배포합니다.
- **Advanced Settings > Secrets** 메뉴에 `.env` 파일의 변수들을 동일하게 추가해야 AI 및 저장 기능이 작동합니다.

## 📁 프로젝트 구조

- `app.py`: 메인 대시보드 및 AI 비즈니스 로직
- `java/answer/`: 로컬 대화 기록 저장 폴더
- `requirements.txt`: 프로젝트 의존성 라이브러리 목록
- `.gitignore`: 환경 변수 및 임시 파일 보호 설정

---

© 2026 QuantumBrief Project. All rights reserved.

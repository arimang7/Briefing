# 💎 QuantumBrief Pro

**QuantumBrief Pro** is an intelligent stock analysis dashboard that combines technical harmonic pattern analysis with Google Gemini AI. It supports real-time market data analysis, consultation with an AI Analyst, and automatic saving of consultation results to a Notion database.

## ✨ Key Features

- **Real-time Technical Analysis**: Automatic detection of Harmonic Patterns (Gartley, Bat, Butterfly, etc.) and RSI indicators.
- **AI Quant Analyst**: Ask questions and get professional answers based on dashboard data using the Gemini 1.5 Flash model.
- **Permanent Data Storage**: Conversation history is automatically saved to both local Markdown files (`java/answer/`) and a **Notion Database**.
- **Global Asset Monitoring**: Integrated monitoring of Korean markets (KOSPI/KOSDAQ), US markets (NASDAQ/NYSE), Bitcoin, and Treasury yields.

## 🛠 Installation & Local Execution

1. **Clone the Repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/Briefing.git
   cd Briefing
   ```

2. **Set up Virtual Environment & Install Packages**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Create a `.env` file in the project root and enter the following format. (Ensure this file is never exposed publicly!)

   ```text
   GEMINI_API_KEY=your_gemini_api_key
   NOTION_TOKEN=your_notion_integration_token
   NOTION_DATABASE_ID=your_notion_database_id
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🚀 Deployment & Auto-save Configuration (Streamlit Cloud & Notion)

### 1. Notion Integration Setup

- Create a new integration at [Notion Developers](https://www.notion.so/my-integrations).
- Create a Notion database to store analysis results, and in the top right `...` -> `Add connections`, allow the created integration.
- Set the first column (Title) name as **"주식 분석"** and the second column as **"날짜"** (Date type).

### 2. Streamlit Cloud Deployment

- Connect your GitHub repository to deploy.
- Add the variables from the `.env` file to the **Advanced Settings > Secrets** menu for AI and storage functions to work.

## 📁 Project Structure

- `app.py`: Main dashboard and AI business logic.
- `java/answer/`: Local folder for saving conversation records.
- `requirements.txt`: List of project dependency libraries.
- `.gitignore`: Protection settings for environment variables and temporary files.

---

© 2026 QuantumBrief Project. All rights reserved.

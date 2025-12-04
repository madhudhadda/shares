# ==============================================================================
# AI FOR INVESTOR - Enhanced Edition (Fundamental + Technical Analysis)
#
# This application provides comprehensive investment analysis by integrating:
# - Fundamental Analysis: Financial statements, valuation metrics, business moats
# - Technical Analysis: Trend, momentum, volatility indicators, and chart patterns
# - AI-Powered Insights: Perplexity AI analyzes both dimensions for holistic recommendations
# ==============================================================================

import os
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import io

# Import technical indicators
import technical_indicators as ti

# Import SaaS modules
import usage_manager as um
import report_generator as rg

# --- Step 1: Page Configuration and API Key Setup ---
st.set_page_config(page_title="AI Holistic Investor (India)", page_icon="📈", layout="wide")
load_dotenv()

# Custom CSS for Modern Purple-Pink Theme
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary-purple: #E6E6FA;  /* Lavender */
        --light-pink: #FFF0F5;      /* Very light pink (Lavender Blush) */
        --accent-purple: #D8BFD8;   /* Thistle */
        --dark-purple: #9370DB;     /* Medium Purple */
        --text-dark: #4A4A4A;
        --card-bg: #FAFAFA;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--light-pink) 100%);
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: var(--dark-purple) !important;
        font-weight: 700 !important;
    }
    
    /* Paragraphs and general text */
    p, span, div, label {
        color: var(--text-dark) !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: var(--text-dark) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E6E6FA 0%, #FFF0F5 100%);
        border-right: 2px solid var(--accent-purple);
    }
    
    [data-testid="stSidebar"] h1 {
        color: var(--dark-purple) !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: var(--text-dark) !important;
    }
    
    /* Input boxes */
    .stTextInput input {
        border: 2px solid var(--accent-purple) !important;
        border-radius: 10px !important;
        padding: 10px !important;
        background-color: white !important;
        color: var(--text-dark) !important;
    }
    
    .stTextInput label {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }
    
    .stTextInput input:focus {
        border-color: var(--dark-purple) !important;
        box-shadow: 0 0 10px rgba(147, 112, 219, 0.3) !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--dark-purple) 0%, #BA55D3 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        box-shadow: 0 4px 15px rgba(147, 112, 219, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(147, 112, 219, 0.6) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(147, 112, 219, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--primary-purple);
        border-radius: 10px;
        color: var(--dark-purple);
        font-weight: 600;
        padding: 10px 20px;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--dark-purple) 0%, #BA55D3 100%);
        color: white !important;
        border: 2px solid var(--dark-purple);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: white !important;
        border: 2px solid var(--accent-purple) !important;
        border-radius: 10px !important;
        color: var(--dark-purple) !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--primary-purple) !important;
        border-color: var(--dark-purple) !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 15px !important;
        border-left: 5px solid var(--dark-purple) !important;
        background-color: white !important;
    }
    
    .stAlert p {
        color: var(--text-dark) !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(147, 112, 219, 0.2) !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: var(--dark-purple) !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-dark) !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-purple) 0%, var(--accent-purple) 50%, var(--light-pink) 100%);
        margin: 30px 0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--dark-purple) !important;
    }
    
    /* Custom card styling */
    .custom-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(147, 112, 219, 0.15);
        border: 2px solid var(--accent-purple);
        margin: 20px 0;
    }
    
    /* Link styling */
    a {
        color: var(--dark-purple) !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    
    a:hover {
        color: #BA55D3 !important;
        text-decoration: underline !important;
    }
    
    /* Ensure all text in content is visible */
    .element-container, .stMarkdown, .stText {
        color: var(--text-dark) !important;
    }

</style>

<div style="text-align: center; padding: 30px; background: white; border-radius: 25px; margin: 20px 0; box-shadow: 0 10px 30px rgba(147, 112, 219, 0.2); border: 3px solid #E6E6FA;">
    <h1 style="color: #9370DB; font-size: 48px; margin: 0; font-weight: 800; background: linear-gradient(135deg, #9370DB 0%, #BA55D3 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📈 AI Holistic Investor</h1>
    <p style="color: #4A4A4A; font-size: 18px; margin: 10px 0 0 0; font-weight: 500;">🇮🇳 India Edition - Powered by Perplexity AI</p>
    <p style="color: #9370DB; font-size: 14px; margin: 5px 0 0 0;">Comprehensive Fundamental + Technical Analysis</p>
</div>
""", unsafe_allow_html=True)

# --- Centralized API Key Configuration ---
st.sidebar.title("🛠️ API Configuration")
perplexity_api_key = st.sidebar.text_input("Enter your Perplexity API Key", value=os.getenv("PERPLEXITY_API_KEY", ""), type="password")

if perplexity_api_key:
    # Initialize OpenAI client for Perplexity
    client = OpenAI(api_key=perplexity_api_key, base_url="https://api.perplexity.ai")
    
    # Initialize and display usage tracking
    um.initialize_usage()
    um.display_usage_widget()
else:
    st.info("Please enter your Perplexity API Key in the sidebar to use the application.")
    st.stop()

# --- Step 2: Enhanced Data Fetching with Technical Indicators ---

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def fetch_deep_stock_data(symbol):
    """Fetches comprehensive data dossier including fundamentals and technicals."""
    try:
        # Auto-append .NS if no suffix is present, assuming Indian NSE stock
        if "." not in symbol:
            symbol = f"{symbol}.NS"
        
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Check if we got valid data
        if not info:
            st.warning(f"No info data returned for {symbol}. The ticker might be invalid.")
            return None
        
        # Verify we have at least some key fields
        if 'symbol' not in info and 'longName' not in info and 'shortName' not in info:
            st.warning(f"Incomplete data for {symbol}. The ticker might be invalid.")
            return None

        # Safely extract news titles
        news_titles = []
        try:
            if stock.news:
                for news_item in stock.news[:5]:
                    # Try different possible keys for the title
                    title = news_item.get('title') or news_item.get('headline') or news_item.get('text', 'No title')
                    news_titles.append(title)
        except Exception as news_error:
            st.warning(f"Could not fetch news for {symbol}: {news_error}")
            news_titles = []
        
        # Fetch historical data (1 year)
        history = stock.history(period="1y")
        
        # Calculate Technical Indicators
        technical_data = calculate_technical_indicators(history)
        
        data_dossier = {
            "symbol": symbol,
            "info": info,
            "history": history,
            "news": news_titles,
            "income_stmt": stock.income_stmt.iloc[:, 0] if not stock.income_stmt.empty else None,
            "balance_sheet": stock.balance_sheet.iloc[:, 0] if not stock.balance_sheet.empty else None,
            "cash_flow": stock.cashflow.iloc[:, 0] if not stock.cashflow.empty else None,
            "technical": technical_data,
        }
        return data_dossier
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def calculate_technical_indicators(ohlcv):
    """Calculate key technical indicators from OHLCV data."""
    if ohlcv.empty:
        return {}
    
    try:
        # Trend Indicators
        sma_50 = ti.SMA(ohlcv, period=50)
        sma_200 = ti.SMA(ohlcv, period=200)
        ema_12 = ti.EMA(ohlcv, period=12)
        ema_26 = ti.EMA(ohlcv, period=26)
        macd = ti.MACD(ohlcv, period_fast=12, period_slow=26, signal=9)
        
        # Momentum Indicators
        rsi = ti.RSI(ohlcv, period=14)
        stoch = ti.STOCH(ohlcv, period=14)
        stochd = ti.STOCHD(ohlcv, period=3, stoch_period=14)
        
        # Volatility Indicators
        bollinger = ti.BOLLINGER(ohlcv, period=20, dev=2)
        atr = ti.ATR(ohlcv, period=14)
        
        # Volume Indicators
        obv = ti.OBV(ohlcv)
        adl = ti.ADL(ohlcv)
        
        # Strength Indicators
        adx = ti.ADX(ohlcv, period=14)
        dmi = ti.DMI(ohlcv, period=14)
        
        # Get latest values for analysis
        latest_close = ohlcv['Close'].iloc[-1]
        latest_rsi = rsi.iloc[-1] if not rsi.empty else None
        latest_macd = macd['MACD'].iloc[-1] if not macd.empty else None
        latest_macd_signal = macd['SIGNAL'].iloc[-1] if not macd.empty else None
        latest_adx = adx.iloc[-1] if not adx.empty else None
        
        # Technical Summary
        technical_summary = {
            "current_price": latest_close,
            "sma_50": sma_50.iloc[-1] if len(sma_50) > 0 else None,
            "sma_200": sma_200.iloc[-1] if len(sma_200) > 0 else None,
            "rsi_14": latest_rsi,
            "macd": latest_macd,
            "macd_signal": latest_macd_signal,
            "adx": latest_adx,
            "bb_upper": bollinger['BB_UPPER'].iloc[-1] if not bollinger.empty else None,
            "bb_lower": bollinger['BB_LOWER'].iloc[-1] if not bollinger.empty else None,
            "atr": atr.iloc[-1] if not atr.empty else None,
        }
        
        return {
            "indicators": {
                "SMA_50": sma_50,
                "SMA_200": sma_200,
                "EMA_12": ema_12,
                "EMA_26": ema_26,
                "MACD": macd,
                "RSI": rsi,
                "STOCH": stoch,
                "STOCHD": stochd,
                "BOLLINGER": bollinger,
                "ATR": atr,
                "OBV": obv,
                "ADL": adl,
                "ADX": adx,
                "DMI": dmi,
            },
            "summary": technical_summary
        }
    except Exception as e:
        st.warning(f"Error calculating technical indicators: {e}")
        return {}

# --- Step 3: AI Analysis Functions ---

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def generate_holistic_investment_report(_stocks_data):
    """Orchestrates AI to generate a comprehensive investment report with fundamental + technical analysis."""
    
    all_dossiers = ""
    for symbol, data in _stocks_data.items():
        if not data: continue
        info = data['info']
        income = data['income_stmt']
        balance = data['balance_sheet']
        cashflow = data['cash_flow']
        technical = data.get('technical', {})
        tech_summary = technical.get('summary', {})
        
        # Safe get for financial data
        def get_safe(data_dict, key):
            return data_dict.get(key, 'N/A') if data_dict is not None else 'N/A'

        dossier = f"""
        ## Data Dossier for {info.get('longName', symbol)} ({symbol})
        
        ### FUNDAMENTAL ANALYSIS
        
        **Company Profile:**
        - **Sector:** {info.get('sector', 'N/A')}
        - **Industry:** {info.get('industry', 'N/A')}
        - **Market Cap:** {info.get('marketCap', 0):,} {info.get('currency', 'INR')}

        **Key Financial Metrics:**
        - **P/E Ratio (Trailing):** {info.get('trailingPE', 'N/A')}
        - **P/E Ratio (Forward):** {info.get('forwardPE', 'N/A')}
        - **PEG Ratio:** {info.get('pegRatio', 'N/A')}
        - **Debt to Equity:** {info.get('debtToEquity', 'N/A')}
        - **Return on Equity:** {info.get('returnOnEquity', 'N/A')}

        **Financial Statement Highlights (Latest Year):**
        - **Total Revenue:** {get_safe(income, 'Total Revenue')}
        - **Net Income:** {get_safe(income, 'Net Income')}
        - **Total Assets:** {get_safe(balance, 'Total Assets')}
        - **Free Cash Flow:** {get_safe(cashflow, 'Free Cash Flow')}

        **Business Summary:**
        {info.get('longBusinessSummary', 'No summary available.')}

        ### TECHNICAL ANALYSIS
        
        **Current Price Action:**
        - **Current Price:** ₹{tech_summary.get('current_price', 'N/A')}
        - **50-Day SMA:** ₹{tech_summary.get('sma_50', 'N/A')}
        - **200-Day SMA:** ₹{tech_summary.get('sma_200', 'N/A')}
        - **Price vs SMA50:** {"Above (Bullish)" if tech_summary.get('current_price', 0) > tech_summary.get('sma_50', 0) else "Below (Bearish)"}
        - **Price vs SMA200:** {"Above (Bullish)" if tech_summary.get('current_price', 0) > tech_summary.get('sma_200', 0) else "Below (Bearish)"}

        **Momentum Indicators:**
        - **RSI (14):** {tech_summary.get('rsi_14', 'N/A')} {"(Overbought >70)" if tech_summary.get('rsi_14', 50) > 70 else "(Oversold <30)" if tech_summary.get('rsi_14', 50) < 30 else "(Neutral)"}
        - **MACD:** {tech_summary.get('macd', 'N/A')}
        - **MACD Signal:** {tech_summary.get('macd_signal', 'N/A')}
        - **MACD Status:** {"Bullish (MACD > Signal)" if tech_summary.get('macd', 0) > tech_summary.get('macd_signal', 0) else "Bearish (MACD < Signal)"}

        **Trend Strength:**
        - **ADX:** {tech_summary.get('adx', 'N/A')} {"(Strong Trend >25)" if tech_summary.get('adx', 0) > 25 else "(Weak Trend)"}

        **Volatility:**
        - **ATR (14):** {tech_summary.get('atr', 'N/A')}
        - **Bollinger Upper:** ₹{tech_summary.get('bb_upper', 'N/A')}
        - **Bollinger Lower:** ₹{tech_summary.get('bb_lower', 'N/A')}

        **Recent News:**
        - {data['news']}
        ---
        """
        all_dossiers += dossier
        # Display the dossier for transparency in an expander
        with st.expander(f"📊 View Complete Data Dossier for {symbol}"):
            st.markdown(dossier)

    system_prompt = """
    You are a Senior Investment Analyst specializing in the Indian Stock Market with expertise in both FUNDAMENTAL and TECHNICAL analysis. Your task is to analyze the provided company 'Data Dossiers' and produce a professional, institutional-grade investment report.
    
    For each company, provide a structured analysis covering these points:
    
    **1. EXECUTIVE SUMMARY:** A brief, high-level overview of the company and your investment thesis.
    
    **2. FUNDAMENTAL ANALYSIS:**
       - **Financial Health:** Analyze profitability, balance sheet strength, and cash flow generation
       - **Valuation:** Is it fairly valued, overvalued, or undervalued based on P/E, PEG, and other metrics?
       - **Competitive Moat:** Assess the company's sustainable competitive advantages in the Indian market
       - **Growth Potential:** Identify catalysts for future growth
       - **Risk Factors:** Key risks that could impact performance
    
    **3. TECHNICAL ANALYSIS:**
       - **Trend Direction:** What is the current trend (uptrend, downtrend, sideways)?
       - **Momentum:** Are momentum indicators (RSI, MACD) confirming the trend?
       - **Support & Resistance:** Identify key price levels based on moving averages and Bollinger Bands
       - **Entry/Exit Points:** Suggest optimal entry points for buyers and exit points for holders
       - **Technical Signals:** Are technical indicators flashing buy, sell, or hold signals?
    
    **4. HOLISTIC VERDICT:**
       - Synthesize both fundamental and technical insights
       - Provide a clear investment recommendation: **Strong Buy, Buy, Hold, Sell, or Strong Sell**
       - Include a suggested time horizon (short-term, medium-term, long-term)
       - Note any conflicts between fundamental and technical signals and how to resolve them
    
    After analyzing all companies individually, conclude your report with:
    - **COMPARATIVE ANALYSIS & FINAL RANKING:** A ranked list comparing the stocks from MOST recommended to LEAST recommended, with justification combining both fundamental strength and technical setup.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here are the data dossiers with both fundamental and technical analysis:\\n\\n{all_dossiers}"}
    ]
    
    with st.spinner("🧠 AI is performing holistic analysis (Fundamental + Technical) with Perplexity... This may take some time."):
        try:
            response = client.chat.completions.create(
                model="sonar-pro", # Using sonar-pro for deep research capabilities
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating report: {e}"

# --- Step 4: Visualization Functions ---

def create_technical_charts(symbol, data):
    """Create comprehensive technical analysis charts."""
    history = data['history']
    technical = data.get('technical', {})
    indicators = technical.get('indicators', {})
    
    # Create subplots: Price + Indicators
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} - Price & Moving Averages', 'MACD', 'RSI', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # 1. Candlestick Chart with Bollinger Bands and Moving Averages
    fig.add_trace(go.Candlestick(
        x=history.index,
        open=history['Open'],
        high=history['High'],
        low=history['Low'],
        close=history['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Add Bollinger Bands
    bb = indicators.get('BOLLINGER', pd.DataFrame())
    if not bb.empty:
        fig.add_trace(go.Scatter(x=history.index, y=bb['BB_UPPER'], name='BB Upper', line=dict(color='rgba(186, 85, 211, 0.6)', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=history.index, y=bb['BB_LOWER'], name='BB Lower', line=dict(color='rgba(186, 85, 211, 0.6)', width=2), fill='tonexty', fillcolor='rgba(230, 230, 250, 0.3)'), row=1, col=1)
    
    # Add Moving Averages
    sma_50 = indicators.get('SMA_50')
    sma_200 = indicators.get('SMA_200')
    if sma_50 is not None and not sma_50.empty:
        fig.add_trace(go.Scatter(x=history.index, y=sma_50, name='SMA 50', line=dict(color='#DA70D6', width=2.5)), row=1, col=1)  # Orchid
    if sma_200 is not None and not sma_200.empty:
        fig.add_trace(go.Scatter(x=history.index, y=sma_200, name='SMA 200', line=dict(color='#9370DB', width=2.5)), row=1, col=1)  # Medium Purple
    
    # 2. MACD
    macd = indicators.get('MACD', pd.DataFrame())
    if not macd.empty:
        fig.add_trace(go.Scatter(x=history.index, y=macd['MACD'], name='MACD', line=dict(color='#9370DB', width=2)), row=2, col=1)  # Medium Purple
        fig.add_trace(go.Scatter(x=history.index, y=macd['SIGNAL'], name='Signal', line=dict(color='#FF69B4', width=2)), row=2, col=1)  # Hot Pink
        fig.add_trace(go.Bar(x=history.index, y=macd['MACD'] - macd['SIGNAL'], name='MACD Histogram', marker_color='rgba(216, 191, 216, 0.6)'), row=2, col=1)  # Thistle
    
    # 3. RSI
    rsi = indicators.get('RSI')
    if rsi is not None and not rsi.empty:
        fig.add_trace(go.Scatter(x=history.index, y=rsi, name='RSI', line=dict(color='#9370DB', width=2.5)), row=3, col=1)  # Medium Purple
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="#FF69B4", line_width=2, row=3, col=1)  # Hot Pink
        fig.add_hline(y=30, line_dash="dash", line_color="#BA55D3", line_width=2, row=3, col=1)  # Medium Orchid
    
    # 4. Volume
    fig.add_trace(go.Bar(x=history.index, y=history['Volume'], name='Volume', marker_color='rgba(230, 230, 250, 0.7)'), row=4, col=1)  # Lavender
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(255,255,255,0.95)',
        plot_bgcolor='rgba(250,250,255,0.3)',
        font=dict(color='#4A4A4A', size=12),
        title_font=dict(color='#9370DB', size=16, family='Arial Black')
    )
    
    fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig

def create_technical_summary_table(technical_data):
    """Create a summary table of technical indicators."""
    summary = technical_data.get('summary', {})
    
    data = {
        "Indicator": [
            "Current Price",
            "50-Day SMA",
            "200-Day SMA",
            "RSI (14)",
            "MACD",
            "MACD Signal",
            "ADX",
            "ATR (14)",
            "BB Upper",
            "BB Lower"
        ],
        "Value": [
            f"₹{summary.get('current_price', 'N/A'):.2f}" if summary.get('current_price') else 'N/A',
            f"₹{summary.get('sma_50', 'N/A'):.2f}" if summary.get('sma_50') else 'N/A',
            f"₹{summary.get('sma_200', 'N/A'):.2f}" if summary.get('sma_200') else 'N/A',
            f"{summary.get('rsi_14', 'N/A'):.2f}" if summary.get('rsi_14') else 'N/A',
            f"{summary.get('macd', 'N/A'):.2f}" if summary.get('macd') else 'N/A',
            f"{summary.get('macd_signal', 'N/A'):.2f}" if summary.get('macd_signal') else 'N/A',
            f"{summary.get('adx', 'N/A'):.2f}" if summary.get('adx') else 'N/A',
            f"{summary.get('atr', 'N/A'):.2f}" if summary.get('atr') else 'N/A',
            f"₹{summary.get('bb_upper', 'N/A'):.2f}" if summary.get('bb_upper') else 'N/A',
            f"₹{summary.get('bb_lower', 'N/A'):.2f}" if summary.get('bb_lower') else 'N/A',
        ],
        "Signal": [
            "—",
            "Bullish" if summary.get('current_price', 0) > summary.get('sma_50', 0) else "Bearish",
            "Bullish" if summary.get('current_price', 0) > summary.get('sma_200', 0) else "Bearish",
            "Overbought" if summary.get('rsi_14', 50) > 70 else "Oversold" if summary.get('rsi_14', 50) < 30 else "Neutral",
            "Bullish" if summary.get('macd', 0) > summary.get('macd_signal', 0) else "Bearish",
            "—",
            "Strong Trend" if summary.get('adx', 0) > 25 else "Weak Trend",
            "—",
            "—",
            "—"
        ]
    }
    
    return pd.DataFrame(data)

# --- Step 5: Streamlit UI ---

st.markdown("""
<div style='background: white; padding: 30px; border-radius: 20px; margin: 20px 0; box-shadow: 0 8px 25px rgba(147, 112, 219, 0.15); border: 2px solid #E6E6FA;'>
    <h2 style='color: #9370DB; margin-top: 0;'>📊 Holistic Investment Analysis</h2>
    <p style='color: #4A4A4A; font-size: 16px;'>Choose analysis mode and enter stock symbols or upload a portfolio CSV to generate a comprehensive report combining fundamental and technical analysis using Perplexity AI.</p>
</div>
""", unsafe_allow_html=True)

# Analysis Mode Selection
analysis_mode = st.radio(
    "📋 Select Analysis Mode",
    ["Single Stock(s)", "Portfolio Analysis (Top 5)"],
    horizontal=True,
    help="Single Stock: Analyze individual stocks. Portfolio: Upload CSV and analyze top 5 holdings."
)

stocks_symbols_list = []

if analysis_mode == "Single Stock(s)":
    input_symbols = st.text_input(
        "🔍 Enter Indian Stock Symbols (up to 5, comma-separated)", 
        "TCS, RELIANCE",
        help="Example: TCS, INFY, RELIANCE, HDFCBANK, WIPRO"
    )
    if input_symbols:
        stocks_symbols_list = [symbol.strip().upper() for symbol in input_symbols.split(",") if symbol.strip()][:5]
else:
    st.markdown("### 📤 Upload Portfolio CSV")
    st.markdown("""
    <div style='background: #F0F8FF; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9370DB;'>
        <p style='margin: 0; color: #4A4A4A;'><strong>CSV Format:</strong> Your CSV should have a 'Symbol' or 'Ticker' column. Optionally include 'Quantity' or 'Weight' columns for sorting.</p>
        <p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>Example: Symbol,Quantity<br/>TCS,100<br/>INFY,50</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload your portfolio CSV file containing stock symbols"
    )
    
    if uploaded_file is not None:
        try:
            portfolio_df = pd.read_csv(uploaded_file)
            
            # Display uploaded portfolio
            st.success(f"✅ Portfolio uploaded successfully! Found {len(portfolio_df)} stocks.")
            with st.expander("📋 View Uploaded Portfolio"):
                st.dataframe(portfolio_df, use_container_width=True)
            
            # Find symbol column (case-insensitive)
            symbol_col = None
            for col in portfolio_df.columns:
                if col.strip().lower() in ['symbol', 'ticker', 'stock', 'name']:
                    symbol_col = col
                    break
            
            if symbol_col is None:
                st.error("❌ Could not find 'Symbol' or 'Ticker' column in CSV. Please check your file format.")
            else:
                # Sort by Quantity or Weight if available
                if 'Quantity' in portfolio_df.columns or 'quantity' in portfolio_df.columns:
                    qty_col = 'Quantity' if 'Quantity' in portfolio_df.columns else 'quantity'
                    portfolio_df = portfolio_df.sort_values(by=qty_col, ascending=False)
                elif 'Weight' in portfolio_df.columns or 'weight' in portfolio_df.columns:
                    wt_col = 'Weight' if 'Weight' in portfolio_df.columns else 'weight'
                    portfolio_df = portfolio_df.sort_values(by=wt_col, ascending=False)
                
                # Get top 5 stocks
                top_stocks = portfolio_df[symbol_col].head(5).tolist()
                stocks_symbols_list = [str(symbol).strip().upper() for symbol in top_stocks if pd.notna(symbol)]
                
                st.info(f"📊 **Top 5 stocks selected for analysis:** {', '.join(stocks_symbols_list)}")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please ensure your CSV has proper format with headers.")

if st.button("🚀 Generate Holistic Analysis Report", key="invest_button"):
    # Check usage limit before proceeding
    if not um.can_perform_analysis():
        um.show_upgrade_message()
        st.stop()
    
    if not stocks_symbols_list:
        st.warning("Please enter at least one stock symbol or upload a portfolio CSV.")
    else:
        # Increment usage counter
        analysis_type = "portfolio" if analysis_mode == "Portfolio Analysis (Top 5)" else "single"
        um.increment_usage(analysis_type)
        
        with st.spinner('Fetching comprehensive data (fundamentals + technicals) for all stocks...'):
            all_stocks_data = {symbol: fetch_deep_stock_data(symbol) for symbol in stocks_symbols_list}
        
        valid_stocks_data = {s: d for s, d in all_stocks_data.items() if d}

        if not valid_stocks_data:
            st.error("Could not retrieve data for any symbols. Please check ticker format.")
        else:
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Analysis", "📈 Technical Charts", "📊 Technical Indicators", "📥 Download Report"])
            
            with tab1:
                st.subheader("🤖 AI-Generated Holistic Investment Report")
                report = generate_holistic_investment_report(valid_stocks_data)
                st.markdown(report)
                
                if analysis_mode == "Portfolio Analysis (Top 5)":
                    st.success("📊 This is a portfolio analysis of your top 5 holdings.")
                
                st.info("💡 This report is AI-generated combining fundamental and technical analysis. For informational purposes only - not financial advice.")
            
            with tab2:
                st.subheader("📈 Technical Analysis Charts")
                for symbol, data in valid_stocks_data.items():
                    st.markdown(f"### {symbol}")
                    tech_chart = create_technical_charts(symbol, data)
                    st.plotly_chart(tech_chart, use_container_width=True)
            
            with tab3:
                st.subheader("📊 Technical Indicators Summary")
                for symbol, data in valid_stocks_data.items():
                    st.markdown(f"### {symbol}")
                    tech_table = create_technical_summary_table(data.get('technical', {}))
                    st.dataframe(tech_table, use_container_width=True, hide_index=True)
                    st.divider()
            
            with tab4:
                st.subheader("📥 Download Analysis Report")
                st.markdown("""
                <div style='background: #F0F8FF; padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #9370DB;'>
                    <p style='margin: 0; color: #4A4A4A;'><strong>Download your comprehensive investment report as a PDF document.</strong></p>
                    <p style='margin: 10px 0 0 0; color: #666; font-size: 14px;'>Includes AI analysis, technical indicators, and stock summaries.</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Generate PDF report
                    pdf_buffer = rg.create_pdf_report(valid_stocks_data, report, "investment_report.pdf")
                    
                    # Download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"AI_Investment_Report_{timestamp}.pdf"
                    
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer,
                        file_name=filename,
                        mime="application/pdf",
                        help="Click to download your investment analysis report"
                    )
                    
                    st.success("✅ Report ready for download!")
                    
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("The analysis is still available in other tabs.")
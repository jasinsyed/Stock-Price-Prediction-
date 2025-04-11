import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import os
import plotly.graph_objects as go
import traceback
import base64

# Load environment variables
load_dotenv()

# Custom CSS for a compact and modern interface
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stSelectbox, .stDateInput {
        background-color: #262730;
        border-radius: 5px;
        padding: 2px;
        font-size: 0.65rem;
    }
    .st-expander {
        background-color: #262730;
        border-radius: 5px;
    }
    .metric {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .chatbot {
        background-color: #1F2937;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border: 2px solid #00FFAA;
        box-shadow: 0 0 10px #00FFAA;
        font-size: 0.9rem;
    }
    .neon-text {
        color: #00FFAA;
        text-shadow: 0 0 5px #00FFAA, 0 0 10px #00FFAA;
    }
    .normal-search .stTextInput input {
         background-color: #fff !important;
         border: 1px solid #ccc !important;
         border-radius: 4px !important;
         padding: 8px !important;
         font-size: 1rem !important;
         color: #000 !important;
    }
    .prediction-alert {
        border-left: 4px solid #00FFAA;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #1a1a1a;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_stock_model():
    try:
        model = load_model("C:\\Users\\Jasin\\OneDrive\\Desktop\\stockprice\\Latest_stock_price_model.keras")
        model.name = "LSTM-v1.2"  # Add model version attribute
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_stock_model()

# Initialize Groq client
@st.cache_resource
def init_groq():
    try:
        return Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception as e:
        st.error("Groq API key not found/invalid")
        st.stop()

groq_client = init_groq()

# NewsAPI Configuration
@st.cache_data
def fetch_market_news(query='stock market', page_size=5):
    try:
        url = f'https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={os.getenv("NEWS_API_KEY")}'
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get('articles', [])
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return []

# Enhanced Chatbot with Neon Theme
def chatbot_response(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main App Title
st.title('📈 AI Stock Market Analyst Pro')

# Main Input Section
with st.container():
    st.markdown('<div class="normal-search">', unsafe_allow_html=True)
    stock = st.text_input('Enter Stock Symbol', 'GOOG').upper()
    st.markdown('</div>', unsafe_allow_html=True)

start_date = st.date_input('Start Date', datetime(2010, 1, 1))
end_date = st.date_input('End Date', datetime.today())

# Sidebar Chatbot
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("NeonAI.png")

with st.sidebar:
    st.header("💬 AI Assistant")
    user_input = st.text_input("Ask me anything about the stock market:")
    if user_input:
        with st.spinner("Analyzing..."):
            response = chatbot_response(user_input)
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start;">
                <img src="data:image/png;base64,{logo_base64}" width="20" style="margin-right: 8px; margin-top: 4px;"/>
                <span style="line-height: 1.5;">{response}</span>
            </div>
            """, unsafe_allow_html=True)

# Stock Data Loading
@st.cache_data
def load_data(stock, start, end):
    try:
        data = yf.download(stock, start, end)
        if data.empty:
            raise ValueError("No data found")
        return data
    except Exception as e:
        st.error(f"Data error: {e}")
        st.stop()

data = load_data(stock, start_date, end_date)

# Real-time Metrics
try:
    info = yf.Ticker(stock).info
    col1, col2, col3, col4 = st.columns(4)
    metric_data = {
        'currentPrice': info.get('currentPrice', 'N/A'),
        'marketCap': info.get('marketCap', 0)/1e9,
        'dayLow': info.get('dayLow', 0),
        'dayHigh': info.get('dayHigh', 0),
        'trailingPE': info.get('trailingPE', 'N/A')
    }
    
    with col1:
        st.markdown(f"<div class='metric'>💰 Current Price<br>${metric_data['currentPrice']:.2f}</div>", 
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric'>📊 Market Cap<br>${metric_data['marketCap']:.2f}B</div>", 
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric'>📈 Day Range<br>{metric_data['dayLow']:.2f} - {metric_data['dayHigh']:.2f}</div>", 
                    unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric'>🔍 PE Ratio<br>{metric_data['trailingPE']:.2f}</div>", 
                    unsafe_allow_html=True)
except Exception as e:
    st.error(f"Info error: {e}")

# Moving Averages Section
st.subheader('📊 Moving Averages')
ma_periods = [50, 100, 200]
for period in ma_periods:
    data[f'MA{period}'] = data['Close'].rolling(period).mean()

selected_ma = st.selectbox('Select Moving Averages to Compare', 
                          ['Price vs MA50', 'MA50 vs MA100', 'MA100 vs MA200'])
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

if selected_ma == 'Price vs MA50':
    ax.plot(data['Close'], label='Closing Price', color='#00FFAA')
    ax.plot(data['MA50'], label='50-Day MA', color='#FFAA00')
elif selected_ma == 'MA50 vs MA100':
    ax.plot(data['MA50'], label='50-Day MA', color='#00FFAA')
    ax.plot(data['MA100'], label='100-Day MA', color='#FFAA00')
else:
    ax.plot(data['MA100'], label='100-Day MA', color='#00FFAA')
    ax.plot(data['MA200'], label='200-Day MA', color='#FFAA00')

ax.set_title(selected_ma, color='white')
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
st.pyplot(fig)

# Enhanced Prediction Section
try:
    # Data validation
    if len(data) < 1000:
        st.warning("⚠️ Limited historical data may affect prediction accuracy")
    
    split_index = int(len(data) * 0.80)
    if split_index < 100:
        raise ValueError("Insufficient data for training - need at least 100 samples")
    
    data_train = pd.DataFrame(data['Close'].iloc[:split_index])
    data_test = pd.DataFrame(data['Close'].iloc[split_index:])
    
    if len(data_test) < 30:
        st.warning("Limited test data - predictions may be less reliable")

    # Data scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)

    lookback = 100
    try:
        test_sequence = pd.concat([data_train.iloc[-lookback:], data_test])
    except Exception as e:
        st.error(f"Data concatenation error: {e}")
        st.stop()

    if len(test_sequence) < lookback + 1:
        raise ValueError(f"Need at least {lookback+1} days of combined data")

    test_scaled = scaler.transform(test_sequence)

    # Create sequences
    X_test, y_test = [], []
    try:
        for i in range(lookback, len(test_scaled)):
            X_test.append(test_scaled[i-lookback:i])
            y_test.append(test_scaled[i])
    except IndexError as e:
        st.error(f"Sequence creation error: {e}")
        st.stop()

    if not X_test:
        raise ValueError("No test sequences created")

    X_test, y_test = np.array(X_test), np.array(y_test)

    # Model validation
    if X_test.ndim != 3 or X_test.shape[1] != lookback:
        raise ValueError(f"Invalid input shape {X_test.shape}")

    # Generate predictions
    with st.spinner('🔮 Generating predictions...'):
        predictions = model.predict(X_test)
        if predictions.size == 0:
            raise ValueError("Empty predictions")

    # Inverse transform
    try:
        predictions = scaler.inverse_transform(predictions).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    except ValueError as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # Date alignment
    plot_dates = data_test.index[:len(y_test)]
    if len(plot_dates) != len(predictions):
        raise ValueError(f"Date/prediction mismatch")

    # Interactive visualization
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
        x=plot_dates,
        y=y_test,
        name='Actual Price',
        line=dict(color='#00FFAA', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    pred_fig.add_trace(go.Scatter(
        x=plot_dates,
        y=predictions,
        name='Predicted Price',
        line=dict(color='#ef553b', width=2, dash='dot'),
        hovertemplate='Date: %{x}<br>Prediction: $%{y:.2f}<extra></extra>'
    ))
    
    # Confidence band
    pred_fig.add_trace(go.Scatter(
        x=plot_dates,
        y=predictions * 1.05,
        fill=None,
        mode='lines',
        line=dict(color='#ef553b', width=0),
        showlegend=False
    ))
    pred_fig.add_trace(go.Scatter(
        x=plot_dates,
        y=predictions * 0.95,
        fill='tonexty',
        mode='lines',
        line=dict(color='#ef553b', width=0),
        fillcolor='rgba(239,85,59,0.2)',
        name='Confidence Band'
    ))

    pred_fig.update_layout(
        height=500,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified',
        title={
            'text': f"{stock} Price Predictions",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        )
    )

    # Display predictions
    st.subheader('🔮 AI Price Predictions')
    
    if not pred_fig.data:
        st.warning("No prediction data available")
        st.stop()
        
    with st.expander("Prediction Details", expanded=True):
        st.markdown(f"""
        - **Prediction Period:** {plot_dates[0].strftime('%Y-%m-%d')} to {plot_dates[-1].strftime('%Y-%m-%d')}
        - **Data Points:** {len(predictions)} predictions
        - **Model Version:** {model.name if hasattr(model, 'name') else '1.0'}
        """)
        
        st.plotly_chart(pred_fig, use_container_width=True)
        st.success(f"Successfully predicted {len(predictions)} trading days")

except Exception as e:
    st.error(f"""
    🚨 Prediction Error: {str(e)}
    
    Common Fixes:
    1. Check stock has sufficient history
    2. Try a different date range
    3. Verify model input format
    """)
    st.error(f"Technical Details:\n{traceback.format_exc()}")
    st.stop()

# News Section
st.subheader('📰 Recent Market News')
news_articles = fetch_market_news(query='stock market')

if news_articles:
    for article in news_articles:
        with st.expander(article.get('title', 'No title')):
            col1, col2 = st.columns([1, 3])
            with col1:
                if article.get('urlToImage'):
                    st.image(
                        article['urlToImage'],
                        width=150,
                        caption=article.get('source', {}).get('name', '')
                    )
            with col2:
                st.write(f"**Source:** {article.get('source', {}).get('name', 'N/A')}")
                if article.get('publishedAt'):
                    st.write(f"**Published:** {pd.to_datetime(article['publishedAt']).strftime('%Y-%m-%d %H:%M:%S')}")
                if article.get('description'):
                    st.write(f"**Description:** {article['description']}")
                st.markdown(f"[Read more]({article.get('url', '#')})")
else:
    st.write("No recent market news available.")
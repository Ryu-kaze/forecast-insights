"""
Predictive Sales Dashboard with Streamlit + Gemini AI
Run with: streamlit run streamlit_sales_dashboard.py
Requirements: pip install streamlit pandas numpy scikit-learn plotly google-generativeai python-dotenv
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Predictive Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0f1729;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a2332 0%, #0f1729 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #14b8a6;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
    }
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    .ai-response {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Gemini
def init_gemini():
    """Initialize Gemini AI with API key"""
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False


# Generate sample sales data
@st.cache_data
def generate_sales_data():
    """Generate realistic sales data with seasonal patterns"""
    np.random.seed(42)
    
    # Date range: 2 years of monthly data
    dates = pd.date_range(start='2023-01-01', periods=24, freq='MS')
    
    # Base sales with seasonal pattern
    base_sales = np.array([
        85000, 78000, 92000, 98000, 105000, 115000,
        108000, 112000, 125000, 135000, 155000, 180000,
        95000, 88000, 102000, 110000, 118000, 128000,
        120000, 125000, 140000, 150000, 170000, 195000
    ])
    
    # Add some random noise
    noise = np.random.normal(0, 5000, len(base_sales))
    actual_sales = base_sales + noise
    
    df = pd.DataFrame({
        'Date': dates,
        'Month': dates.strftime('%b %Y'),
        'Actual_Sales': actual_sales.astype(int),
        'Month_Num': range(1, 25),
        'Season': [get_season(d.month) for d in dates],
        'Quarter': [f'Q{(d.month-1)//3 + 1}' for d in dates],
        'Year': dates.year
    })
    
    return df


def get_season(month):
    """Get season from month number"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


# ML Model for predictions
@st.cache_resource
def train_model(df, model_type='RandomForest'):
    """Train ML model for sales prediction"""
    # Feature engineering
    X = df[['Month_Num']].values
    y = df['Actual_Sales'].values
    
    # Add seasonal features
    df_features = df.copy()
    df_features['Month_of_Year'] = df_features['Date'].dt.month
    df_features['Is_Holiday_Season'] = df_features['Month_of_Year'].isin([11, 12]).astype(int)
    
    X_extended = np.column_stack([
        df_features['Month_Num'],
        df_features['Month_of_Year'],
        df_features['Is_Holiday_Season']
    ])
    
    if model_type == 'LinearRegression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_extended, y)
    
    # Predictions for existing data
    predictions = model.predict(X_extended)
    
    return model, predictions


def predict_future(model, df, months_ahead=6):
    """Predict future sales"""
    last_month_num = df['Month_Num'].max()
    last_date = df['Date'].max()
    
    future_dates = pd.date_range(start=last_date + timedelta(days=32), periods=months_ahead, freq='MS')
    
    future_features = []
    for i, date in enumerate(future_dates):
        month_num = last_month_num + i + 1
        month_of_year = date.month
        is_holiday = 1 if month_of_year in [11, 12] else 0
        future_features.append([month_num, month_of_year, is_holiday])
    
    future_predictions = model.predict(np.array(future_features))
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Month': future_dates.strftime('%b %Y'),
        'Predicted_Sales': future_predictions.astype(int),
        'Is_Future': True
    })
    
    return future_df


def get_gemini_insights(df, future_df, kpis):
    """Get AI-powered insights from Gemini"""
    if not init_gemini():
        return "⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file or Streamlit secrets."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare context
        prompt = f"""
        Analyze this sales data and provide strategic insights:
        
        **Historical Performance (Last 24 months):**
        - Total Actual Sales: ${kpis['total_actual']:,.0f}
        - Total Predicted Sales: ${kpis['total_predicted']:,.0f}
        - Overall Variance: ${kpis['variance']:,.0f} ({kpis['variance_pct']:.1f}%)
        - Model Accuracy (MAPE): {kpis['mape']:.2f}%
        - Forecast Accuracy: {kpis['accuracy']:.1f}%
        
        **Recent Trend (Last 6 months):**
        {df.tail(6)[['Month', 'Actual_Sales']].to_string(index=False)}
        
        **AI Predicted Future Sales (Next 6 months):**
        {future_df[['Month', 'Predicted_Sales']].to_string(index=False)}
        
        Provide:
        1. Key insights about sales performance
        2. Seasonal patterns identified
        3. Risk factors to watch
        4. Actionable recommendations for improving sales
        5. Confidence assessment of future predictions
        
        Keep response concise and actionable (max 300 words).
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ Error getting AI insights: {str(e)}"


def calculate_dax_metrics(df, predictions):
    """Calculate DAX-equivalent metrics"""
    variance = df['Actual_Sales'].sum() - predictions.sum()
    variance_pct = (variance / predictions.sum()) * 100
    mape = mean_absolute_percentage_error(df['Actual_Sales'], predictions) * 100
    accuracy = 100 - mape
    
    return {
        'total_actual': df['Actual_Sales'].sum(),
        'total_predicted': predictions.sum(),
        'variance': variance,
        'variance_pct': variance_pct,
        'mape': mape,
        'accuracy': accuracy
    }


# Main App
def main():
    # Header
    st.markdown("# 📊 Predictive Sales Dashboard")
    st.markdown("*ML-powered forecasting with Scikit-learn + Gemini AI insights*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        model_type = st.selectbox(
            "ML Model",
            ["RandomForest", "LinearRegression"],
            help="Select the machine learning model for predictions"
        )
        
        future_months = st.slider(
            "Forecast Months Ahead",
            min_value=3,
            max_value=12,
            value=6,
            help="Number of months to predict into the future"
        )
        
        st.markdown("---")
        st.markdown("### 🔑 Gemini API")
        
        # API Key input (if not in env)
        if not os.getenv("GEMINI_API_KEY"):
            api_key = st.text_input(
                "Enter Gemini API Key",
                type="password",
                help="Get your API key from https://makersuite.google.com/app/apikey"
            )
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
                st.success("✅ API Key configured!")
        else:
            st.success("✅ API Key loaded from environment")
        
        st.markdown("---")
        st.markdown("### 📥 Export")
        
    # Load and process data
    df = generate_sales_data()
    model, predictions = train_model(df, model_type)
    future_df = predict_future(model, df, future_months)
    kpis = calculate_dax_metrics(df, predictions)
    
    # Add predictions to main df
    df['Predicted_Sales'] = predictions.astype(int)
    df['Variance'] = df['Actual_Sales'] - df['Predicted_Sales']
    df['Variance_Pct'] = (df['Variance'] / df['Predicted_Sales']) * 100
    
    # KPI Cards Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Actual Sales",
            f"${kpis['total_actual']/1e6:.2f}M",
            "Total Revenue"
        )
    
    with col2:
        st.metric(
            "📈 Predicted Sales",
            f"${kpis['total_predicted']/1e6:.2f}M",
            "ML Forecast"
        )
    
    with col3:
        delta_color = "normal" if kpis['variance'] >= 0 else "inverse"
        st.metric(
            "📊 Variance",
            f"${abs(kpis['variance'])/1e3:.0f}K",
            f"{kpis['variance_pct']:+.1f}%",
            delta_color=delta_color
        )
    
    with col4:
        st.metric(
            "🎯 Forecast Accuracy",
            f"{kpis['accuracy']:.1f}%",
            "Model Performance"
        )
    
    with col5:
        st.metric(
            "📉 MAPE",
            f"{kpis['mape']:.2f}%",
            "Mean Absolute % Error"
        )
    
    st.markdown("---")
    
    # Main Chart
    st.markdown("### 📈 Sales Forecast Analysis")
    
    # Create combined dataframe for chart
    chart_df = df[['Date', 'Month', 'Actual_Sales', 'Predicted_Sales']].copy()
    chart_df['Type'] = 'Historical'
    
    future_chart = future_df.copy()
    future_chart['Actual_Sales'] = None
    future_chart['Type'] = 'Forecast'
    future_chart = future_chart.rename(columns={'Predicted_Sales': 'Predicted_Sales'})
    
    # Plotly chart
    fig = go.Figure()
    
    # Actual sales line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual_Sales'],
        name='Actual Sales',
        line=dict(color='#14b8a6', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    # Predicted sales line (historical)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Predicted_Sales'],
        name='Predicted (Historical)',
        line=dict(color='#f59e0b', width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_df['Date'],
        y=future_df['Predicted_Sales'],
        name='Future Forecast',
        line=dict(color='#3b82f6', width=3, dash='dot'),
        mode='lines+markers',
        marker=dict(size=10, symbol='diamond')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 41, 0)',
        plot_bgcolor='rgba(26, 35, 50, 0.5)',
        height=450,
        legend=dict(orientation='h', y=1.1),
        xaxis=dict(gridcolor='rgba(45, 55, 72, 0.5)'),
        yaxis=dict(gridcolor='rgba(45, 55, 72, 0.5)', tickformat='$,.0f')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Two column layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📋 Variance Analysis Table")
        
        # Display table
        display_df = df[['Month', 'Actual_Sales', 'Predicted_Sales', 'Variance', 'Variance_Pct']].copy()
        display_df['Status'] = display_df['Variance'].apply(
            lambda x: '🟢 Above' if x > 0 else ('🔴 Below' if x < 0 else '⚪ On Target')
        )
        display_df['Variance_Pct'] = display_df['Variance_Pct'].apply(lambda x: f"{x:+.2f}%")
        display_df['Actual_Sales'] = display_df['Actual_Sales'].apply(lambda x: f"${x:,}")
        display_df['Predicted_Sales'] = display_df['Predicted_Sales'].apply(lambda x: f"${x:,}")
        display_df['Variance'] = display_df['Variance'].apply(lambda x: f"${x:+,}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    
    with col_right:
        st.markdown("### 📝 DAX Formulas")
        st.markdown("*Power BI Ready Measures*")
        
        dax_formulas = {
            "Variance": "SUM(Sales[Actual]) - SUM(Sales[Predicted])",
            "Variance %": "DIVIDE([Variance], SUM(Sales[Predicted]), 0) * 100",
            "MAPE": "AVERAGEX(Sales, ABS(Sales[Actual] - Sales[Predicted]) / Sales[Actual]) * 100",
            "Forecast Accuracy": "100 - [MAPE]",
            "YTD Actual": "TOTALYTD(SUM(Sales[Actual]), Dates[Date])",
        }
        
        for name, formula in dax_formulas.items():
            with st.expander(f"📌 {name}"):
                st.code(formula, language="dax")
    
    st.markdown("---")
    
    # Gemini AI Insights
    st.markdown("### 🤖 Gemini AI Insights")
    
    if st.button("🔮 Generate AI Predictions & Insights", type="primary"):
        with st.spinner("Analyzing data with Gemini AI..."):
            insights = get_gemini_insights(df, future_df, kpis)
            st.markdown(f"""
            <div class="ai-response">
            {insights}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Future Predictions Table
    st.markdown("### 🔮 Future Sales Predictions")
    
    future_display = future_df[['Month', 'Predicted_Sales']].copy()
    future_display['Predicted_Sales'] = future_display['Predicted_Sales'].apply(lambda x: f"${x:,}")
    future_display['Confidence'] = ['High', 'High', 'Medium', 'Medium', 'Low', 'Low'][:len(future_display)]
    
    st.dataframe(future_display, use_container_width=True)
    
    # Export section in sidebar
    with st.sidebar:
        # Export to CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Historical Data (CSV)",
            csv,
            "sales_data.csv",
            "text/csv",
            use_container_width=True
        )
        
        future_csv = future_df.to_csv(index=False)
        st.download_button(
            "📥 Download Forecast (CSV)",
            future_csv,
            "sales_forecast.csv",
            "text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()

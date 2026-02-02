# Predictive Sales Dashboard

A Streamlit-based sales forecasting dashboard with Scikit-learn ML models and Gemini AI insights.

## Features

- 📊 **ML-Powered Predictions**: Random Forest & Linear Regression models
- 🤖 **Gemini AI Integration**: Get strategic insights and recommendations
- 📈 **Interactive Charts**: Plotly visualizations with dark theme
- 📋 **Variance Analysis**: DAX-equivalent calculations for Power BI
- 📥 **CSV Export**: Export data for Power BI integration

## Quick Start

### 1. Install Dependencies

```bash
cd python_app
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

**Option A**: Create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

**Option B**: Enter directly in the app sidebar

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the App

```bash
streamlit run streamlit_sales_dashboard.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Select ML Model**: Choose between RandomForest or LinearRegression
2. **Set Forecast Period**: Adjust how many months to predict
3. **View KPIs**: See key metrics (Variance, MAPE, Accuracy)
4. **Generate AI Insights**: Click the button to get Gemini-powered analysis
5. **Export Data**: Download CSV files for Power BI

## DAX Formulas (Power BI)

The app includes ready-to-use DAX formulas:

```dax
Variance = SUM(Sales[Actual]) - SUM(Sales[Predicted])
Variance % = DIVIDE([Variance], SUM(Sales[Predicted]), 0) * 100
MAPE = AVERAGEX(Sales, ABS(Sales[Actual] - Sales[Predicted]) / Sales[Actual]) * 100
Forecast Accuracy = 100 - [MAPE]
```

## Project Structure

```
python_app/
├── streamlit_sales_dashboard.py  # Main application
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

## Screenshots

The dashboard features:
- Dark analytics theme
- 5 KPI cards (Actual, Predicted, Variance, Accuracy, MAPE)
- Interactive line chart with actual vs predicted sales
- Variance analysis table with status indicators
- Gemini AI insights panel
- Future predictions table

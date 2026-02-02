"""
Predictive Sales Dashboard - Flask App
Run with: python app.py
Requirements: pip install -r requirements.txt
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Currency conversion rate (approximate)
USD_TO_INR = 83.0

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


def generate_sales_data():
    """Generate realistic sales data ending at current date"""
    np.random.seed(42)
    
    # End at current month, go back 24 months
    end_date = datetime.now().replace(day=1)
    dates = pd.date_range(end=end_date, periods=24, freq='MS')
    
    # Base sales with seasonal pattern (repeating 12-month pattern)
    base_pattern = np.array([
        85000, 78000, 92000, 98000, 105000, 115000,
        108000, 112000, 125000, 135000, 155000, 180000
    ])
    
    # Repeat pattern for 24 months with growth
    base_sales = np.concatenate([base_pattern, base_pattern * 1.1])
    
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


def train_model(df, model_type='RandomForest'):
    """Train ML model for sales prediction"""
    df_features = df.copy()
    df_features['Month_of_Year'] = df_features['Date'].dt.month
    df_features['Is_Holiday_Season'] = df_features['Month_of_Year'].isin([11, 12]).astype(int)
    
    X = np.column_stack([
        df_features['Month_Num'],
        df_features['Month_of_Year'],
        df_features['Is_Holiday_Season']
    ])
    y = df['Actual_Sales'].values
    
    if model_type == 'LinearRegression':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    return model, predictions, df_features


def predict_future(model, df, df_features, months_ahead=6):
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
        'Predicted_Sales': future_predictions.astype(int)
    })
    
    return future_df


def calculate_metrics(df, predictions):
    """Calculate DAX-equivalent metrics"""
    variance = df['Actual_Sales'].sum() - predictions.sum()
    variance_pct = (variance / predictions.sum()) * 100
    mape = mean_absolute_percentage_error(df['Actual_Sales'], predictions) * 100
    accuracy = 100 - mape
    
    return {
        'total_actual': float(df['Actual_Sales'].sum()),
        'total_predicted': float(predictions.sum()),
        'variance': float(variance),
        'variance_pct': float(variance_pct),
        'mape': float(mape),
        'accuracy': float(accuracy)
    }


def get_gemini_insights(df, future_df, kpis, currency='USD'):
    """Get AI-powered insights from Gemini"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        currency_symbol = '₹' if currency == 'INR' else '$'
        multiplier = USD_TO_INR if currency == 'INR' else 1
        
        prompt = f"""
        Analyze this sales data and provide strategic insights:
        
        **Historical Performance (Last 24 months):**
        - Total Actual Sales: {currency_symbol}{kpis['total_actual'] * multiplier:,.0f}
        - Total Predicted Sales: {currency_symbol}{kpis['total_predicted'] * multiplier:,.0f}
        - Overall Variance: {currency_symbol}{kpis['variance'] * multiplier:,.0f} ({kpis['variance_pct']:.1f}%)
        - Model Accuracy (MAPE): {kpis['mape']:.2f}%
        - Forecast Accuracy: {kpis['accuracy']:.1f}%
        
        **Recent Trend (Last 6 months):**
        {df.tail(6)[['Month', 'Actual_Sales']].to_string(index=False)}
        
        **AI Predicted Future Sales (Next 6 months):**
        {future_df[['Month', 'Predicted_Sales']].to_string(index=False)}
        
        Current Date: {datetime.now().strftime('%B %Y')}
        
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


# Store data in memory (for demo purposes)
app_data = {
    'df': None,
    'model': None,
    'predictions': None,
    'future_df': None,
    'kpis': None,
    'df_features': None
}


def initialize_data(custom_df=None):
    """Initialize or reinitialize data"""
    if custom_df is not None:
        app_data['df'] = custom_df
    else:
        app_data['df'] = generate_sales_data()
    
    app_data['model'], app_data['predictions'], app_data['df_features'] = train_model(app_data['df'])
    app_data['future_df'] = predict_future(app_data['model'], app_data['df'], app_data['df_features'])
    app_data['kpis'] = calculate_metrics(app_data['df'], app_data['predictions'])


# Initialize with sample data on startup
initialize_data()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    """Get all dashboard data"""
    currency = request.args.get('currency', 'USD')
    multiplier = USD_TO_INR if currency == 'INR' else 1
    
    df = app_data['df'].copy()
    df['Predicted_Sales'] = app_data['predictions'].astype(int)
    df['Variance'] = df['Actual_Sales'] - df['Predicted_Sales']
    df['Variance_Pct'] = (df['Variance'] / df['Predicted_Sales']) * 100
    
    # Convert to selected currency
    df['Actual_Sales_Display'] = (df['Actual_Sales'] * multiplier).astype(int)
    df['Predicted_Sales_Display'] = (df['Predicted_Sales'] * multiplier).astype(int)
    df['Variance_Display'] = (df['Variance'] * multiplier).astype(int)
    
    future_df = app_data['future_df'].copy()
    future_df['Predicted_Sales_Display'] = (future_df['Predicted_Sales'] * multiplier).astype(int)
    
    kpis = {
        'total_actual': app_data['kpis']['total_actual'] * multiplier,
        'total_predicted': app_data['kpis']['total_predicted'] * multiplier,
        'variance': app_data['kpis']['variance'] * multiplier,
        'variance_pct': app_data['kpis']['variance_pct'],
        'mape': app_data['kpis']['mape'],
        'accuracy': app_data['kpis']['accuracy']
    }
    
    return jsonify({
        'historical': df[['Month', 'Actual_Sales_Display', 'Predicted_Sales_Display', 'Variance_Display', 'Variance_Pct']].to_dict('records'),
        'future': future_df[['Month', 'Predicted_Sales_Display']].to_dict('records'),
        'kpis': kpis,
        'currency': currency,
        'currency_symbol': '₹' if currency == 'INR' else '$'
    })


@app.route('/api/insights')
def get_insights():
    """Get Gemini AI insights"""
    currency = request.args.get('currency', 'USD')
    insights = get_gemini_insights(app_data['df'], app_data['future_df'], app_data['kpis'], currency)
    return jsonify({'insights': insights})


@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Upload custom sales data CSV"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        df = pd.read_csv(file)
        
        # Validate required columns
        required_cols = ['Date', 'Actual_Sales']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': f'CSV must contain columns: {required_cols}'}), 400
        
        # Process the data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['Month'] = df['Date'].dt.strftime('%b %Y')
        df['Month_Num'] = range(1, len(df) + 1)
        df['Season'] = [get_season(d.month) for d in df['Date']]
        df['Quarter'] = [f'Q{(d.month-1)//3 + 1}' for d in df['Date']]
        df['Year'] = df['Date'].dt.year
        
        # Reinitialize with new data
        initialize_data(df)
        
        return jsonify({'success': True, 'message': f'Loaded {len(df)} records'})
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 400


@app.route('/api/export/<data_type>')
def export_data(data_type):
    """Export data as CSV"""
    currency = request.args.get('currency', 'USD')
    multiplier = USD_TO_INR if currency == 'INR' else 1
    
    if data_type == 'historical':
        df = app_data['df'].copy()
        df['Predicted_Sales'] = app_data['predictions'].astype(int)
        df['Variance'] = df['Actual_Sales'] - df['Predicted_Sales']
        
        if currency == 'INR':
            df['Actual_Sales'] = (df['Actual_Sales'] * multiplier).astype(int)
            df['Predicted_Sales'] = (df['Predicted_Sales'] * multiplier).astype(int)
            df['Variance'] = (df['Variance'] * multiplier).astype(int)
        
        export_df = df[['Month', 'Actual_Sales', 'Predicted_Sales', 'Variance']]
        filename = f'sales_historical_{currency}.csv'
        
    elif data_type == 'forecast':
        export_df = app_data['future_df'].copy()
        if currency == 'INR':
            export_df['Predicted_Sales'] = (export_df['Predicted_Sales'] * multiplier).astype(int)
        export_df = export_df[['Month', 'Predicted_Sales']]
        filename = f'sales_forecast_{currency}.csv'
    else:
        return jsonify({'error': 'Invalid export type'}), 400
    
    output = io.StringIO()
    export_df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


@app.route('/api/reset')
def reset_data():
    """Reset to sample data"""
    initialize_data()
    return jsonify({'success': True, 'message': 'Data reset to sample'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

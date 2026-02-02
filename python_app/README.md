# Predictive Sales Dashboard - Flask App

A Flask-based sales forecasting dashboard with Scikit-learn ML models and Gemini AI insights. Deployable from VS Code to any platform.

## Features

- 📊 **ML-Powered Predictions**: Random Forest & Linear Regression models
- 🤖 **Gemini AI Integration**: Strategic insights and recommendations
- 💱 **USD/INR Currency Toggle**: Switch between currencies instantly
- 📁 **Import Custom Data**: Upload your own CSV sales data
- 📥 **Export to Power BI**: Download CSV files for integration
- 🌐 **VS Code Deployable**: Deploy to Render, Railway, Heroku, etc.

## Quick Start

### 1. Install Dependencies

```bash
cd python_app
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run Locally

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## Importing Custom Data

Upload a CSV file with the following format:

| Date | Actual_Sales |
|------|--------------|
| 2024-01-01 | 85000 |
| 2024-02-01 | 78000 |
| 2024-03-01 | 92000 |

The app will automatically:
- Parse the dates
- Generate ML predictions
- Calculate variances and KPIs

## Deployment

### Deploy to Render

1. Create a `render.yaml` file (already included)
2. Push to GitHub
3. Connect to Render and deploy

### Deploy to Railway

1. Push to GitHub
2. Connect to Railway
3. Set `GEMINI_API_KEY` environment variable
4. Deploy

### Deploy to Heroku

```bash
# Create Procfile (already included)
heroku create your-app-name
heroku config:set GEMINI_API_KEY=your-key
git push heroku main
```

### Deploy from VS Code

1. Install Azure/Railway/Render extension
2. Right-click project → Deploy
3. Set environment variables in dashboard

## Project Structure

```
python_app/
├── app.py                 # Flask application
├── templates/
│   └── index.html         # Frontend UI
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── Procfile              # Heroku deployment
├── render.yaml           # Render deployment
└── README.md             # This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/api/data` | GET | Get all dashboard data |
| `/api/insights` | GET | Generate Gemini AI insights |
| `/api/upload` | POST | Upload custom CSV data |
| `/api/export/<type>` | GET | Export data as CSV |
| `/api/reset` | GET | Reset to sample data |

## DAX Formulas (Power BI)

```dax
Variance = SUM(Sales[Actual]) - SUM(Sales[Predicted])
Variance % = DIVIDE([Variance], SUM(Sales[Predicted]), 0) * 100
MAPE = AVERAGEX(Sales, ABS(Sales[Actual] - Sales[Predicted]) / Sales[Actual]) * 100
Forecast Accuracy = 100 - [MAPE]
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `PORT` | Server port (default: 5000) |

## Screenshots

The dashboard includes:
- Dark analytics theme
- 5 KPI cards with currency toggle
- Interactive Plotly chart
- Variance analysis table
- Gemini AI insights panel
- CSV import/export functionality

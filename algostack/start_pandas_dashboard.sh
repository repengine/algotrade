#!/bin/bash

# AlgoStack Pandas Dashboard Launcher
# This version uses pure pandas indicators (no TA-Lib required)

echo "🚀 Starting AlgoStack Dashboard (Pandas Edition)..."
echo "📊 Using pure pandas indicators - no TA-Lib required!"

# Check if we're in the algostack directory
if [ ! -f "dashboard_pandas.py" ]; then
    echo "❌ Error: dashboard_pandas.py not found. Please run from the algostack directory."
    exit 1
fi

# Check for Alpha Vantage API key
if [ -n "$ALPHA_VANTAGE_API_KEY" ]; then
    echo "✅ Alpha Vantage API key detected"
    if [ "$ALPHA_VANTAGE_PREMIUM" = "true" ]; then
        echo "   Premium tier enabled (75 requests/min)"
    else
        echo "   Free tier (5 requests/min)"
    fi
else
    echo "ℹ️  No Alpha Vantage API key found"
    echo "   Set ALPHA_VANTAGE_API_KEY to enable intraday data"
    echo "   Dashboard will use Yahoo Finance for daily data"
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "🔧 Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check for required packages
echo "📦 Checking dependencies..."
python -c "import pandas, numpy, yfinance, streamlit, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing dependencies. Installing..."
    pip install pandas numpy yfinance streamlit plotly
fi

# Start the dashboard
echo ""
echo "🌐 Starting Streamlit dashboard..."
echo "📊 Dashboard will open in your browser automatically"
echo ""

# Run streamlit
streamlit run dashboard_pandas.py --server.port 8501 --server.address localhost
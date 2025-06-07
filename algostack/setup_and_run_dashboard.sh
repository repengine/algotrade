#!/bin/bash

echo "🚀 AlgoStack Dashboard Setup & Run"
echo "=================================="
echo ""

# Navigate to algostack directory
cd "$(dirname "$0")"

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Installing required packages..."
    python3 -m pip install --upgrade pip
    python3 -m pip install streamlit plotly pandas numpy pyyaml
    echo ""
fi

# Get network info
WSL_IP=$(hostname -I | awk '{print $1}')

echo "Starting dashboard..."
echo ""
echo "Try these URLs in your browser:"
echo "  1. http://localhost:8501"
echo "  2. http://127.0.0.1:8501"
echo "  3. http://$WSL_IP:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit
python3 -m streamlit run dashboard_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
#!/bin/bash

# Direct run method for AlgoStack Dashboard

echo "🚀 Starting AlgoStack Dashboard (Direct Method)..."
echo "================================================"

cd "$(dirname "$0")"

# Run streamlit directly with Python
python3 -m streamlit run dashboard_final_integrated.py --server.port=8501 --server.address=localhost
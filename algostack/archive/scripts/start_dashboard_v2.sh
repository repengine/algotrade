#!/bin/bash

# Start the V2 AlgoStack Dashboard with validation bypass

echo "🚀 Starting AlgoStack Dashboard V2..."
echo "===================================="

# Navigate to the algostack directory
cd "$(dirname "$0")"

echo ""
echo "📊 Launching dashboard with enhanced features..."
echo "✨ Features:"
echo "   - Validation bypass for flexibility"
echo "   - Automatic type conversion"
echo "   - Better error handling"
echo "   - All parameters displayed simply"
echo ""
echo "🌐 Opening in your browser..."

python3 -m streamlit run dashboard_production_v2.py --server.port=8501 --server.address=localhost
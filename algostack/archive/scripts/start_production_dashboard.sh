#!/bin/bash

# Start the Production AlgoStack Dashboard

echo "🚀 Starting AlgoStack Production Dashboard..."
echo "==========================================="

# Navigate to the algostack directory
cd "$(dirname "$0")"

# Run using Python module method (most reliable)
echo ""
echo "📊 Launching production dashboard..."
echo "✨ Features:"
echo "   - Complete parameter validation for all strategies"
echo "   - Proper default values for all required parameters"
echo "   - Handles all strategy types correctly"
echo "   - Comprehensive error handling"
echo ""
echo "🌐 Opening in your browser..."

python3 -m streamlit run dashboard_production.py --server.port=8501 --server.address=localhost
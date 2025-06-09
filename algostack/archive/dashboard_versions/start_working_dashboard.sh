#!/bin/bash

echo "🚀 Starting AlgoStack Working Dashboard"
echo "===================================="

cd "$(dirname "$0")"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Get IP addresses
WSL_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "📡 Network Information:"
echo "   WSL IP: $WSL_IP"
echo ""
echo "🌐 Try these URLs in your Windows browser:"
echo "   1. http://localhost:8505"
echo "   2. http://127.0.0.1:8505"  
echo "   3. http://$WSL_IP:8505"
echo ""
echo "Press Ctrl+C to stop"
echo "===================================="
echo ""

# Kill any old dashboards running on this port
pkill -f "streamlit.*8505" 2>/dev/null

# Run the working dashboard on port 8505
exec streamlit run dashboard_working.py \
    --server.port 8505 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
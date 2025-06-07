#!/bin/bash

echo "🚀 Starting AlgoStack Integrated Dashboard"
echo "======================================="

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
echo "   1. http://localhost:8502"
echo "   2. http://127.0.0.1:8502"  
echo "   3. http://$WSL_IP:8502"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================="
echo ""

# Kill the old dashboard if running
pkill -f "streamlit.*dashboard_app_simple.py" 2>/dev/null

# Run the integrated dashboard on a different port
exec streamlit run dashboard_integrated.py \
    --server.port 8502 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
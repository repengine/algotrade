#!/bin/bash

echo "🚀 Starting AlgoStack Simple Dashboard"
echo "===================================="

cd "$(dirname "$0")"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check dependencies
echo "Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "Installing streamlit..."
    pip install streamlit
}

# Get IP addresses
WSL_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "📡 Network Information:"
echo "   WSL IP: $WSL_IP"
echo ""
echo "🌐 Try these URLs in your Windows browser:"
echo "   1. http://localhost:8501"
echo "   2. http://127.0.0.1:8501"  
echo "   3. http://$WSL_IP:8501"
echo ""
echo "💡 If none work, try:"
echo "   - Disable Windows Firewall temporarily"
echo "   - Run in PowerShell as Admin:"
echo "     netsh interface portproxy add v4tov4 listenport=8501 connectport=8501 connectaddress=$WSL_IP"
echo ""
echo "Press Ctrl+C to stop"
echo "===================================="
echo ""

# Run streamlit
exec streamlit run dashboard_app_simple.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
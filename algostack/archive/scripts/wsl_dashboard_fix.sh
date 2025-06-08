#!/bin/bash

echo "🔧 WSL Dashboard Network Configuration"
echo "====================================="
echo ""

# Get network information
WSL_IP=$(hostname -I | awk '{print $1}')
WINDOWS_HOST_IP=$(ip route show | grep default | awk '{print $3}')

echo "Network Information:"
echo "  WSL IP: $WSL_IP"
echo "  Windows Host IP: $WINDOWS_HOST_IP"
echo ""

# Function to test port accessibility
test_port() {
    local port=$1
    echo -n "Testing port $port... "
    if nc -z -v -w1 localhost $port 2>/dev/null; then
        echo "✓ Open"
        return 0
    else
        echo "✗ Closed"
        return 1
    fi
}

# Test common ports
echo "Port Status:"
test_port 8501

echo ""
echo "Starting Streamlit with multiple binding options..."
echo ""

# Start streamlit with explicit network configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Change to app directory
cd /home/republic/algotrade/algostack

echo "Dashboard URLs to try:"
echo "  1. http://localhost:8501"
echo "  2. http://127.0.0.1:8501"
echo "  3. http://$WSL_IP:8501"
echo ""
echo "If none work, run this in Windows PowerShell (as Administrator):"
echo "  netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=$WSL_IP"
echo ""
echo "Starting dashboard..."
echo ""

# Run streamlit
python3 -m streamlit run dashboard_app.py
#!/bin/bash

# Quick installation script for AlgoStack Dashboard
# This skips problematic dependencies and gets you running quickly

echo "🚀 AlgoStack Quick Installation"
echo "==============================="
echo ""
echo "This will install the minimum requirements to run the dashboard."
echo "Technical indicators will use fallback implementations."
echo ""

# Check if we should use --break-system-packages
if pip install --help | grep -q "break-system-packages"; then
    PIP_FLAGS="--break-system-packages"
    echo "Note: Using --break-system-packages flag"
else
    PIP_FLAGS=""
fi

# Install core requirements
echo "📦 Installing core packages..."
pip install $PIP_FLAGS pandas numpy yfinance

echo "📊 Installing visualization packages..."
pip install $PIP_FLAGS streamlit plotly matplotlib seaborn

echo "🔧 Installing additional packages..."
pip install $PIP_FLAGS scipy requests pyyaml click rich

echo ""
echo "✅ Installation complete!"
echo ""
echo "The dashboard is ready to use with fallback technical indicators."
echo "To start the dashboard, run:"
echo "    ./start_final_dashboard.sh"
echo ""
echo "Note: Some advanced features may be limited without the full requirements."
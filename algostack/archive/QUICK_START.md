# AlgoStack Dashboard - Quick Start Guide

## 🚨 First Time Setup

Run this command to set up and start the dashboard:

```bash
cd /home/republic/algotrade/algostack
./setup_and_run_dashboard.sh
```

This script will:
1. Create/activate a virtual environment
2. Install required packages (streamlit, plotly, etc.)
3. Start the dashboard

## 🧪 Test WSL Networking First

If the dashboard doesn't open, test your WSL networking:

```bash
cd /home/republic/algotrade/algostack
python3 test_simple_server.py
```

Then try opening http://localhost:8000 in your browser.

## 🔧 Manual Setup (if script fails)

```bash
cd /home/republic/algotrade/algostack

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install minimal requirements for dashboard
pip install streamlit plotly pandas numpy pyyaml

# Run dashboard
streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🌐 URL Options

Try these URLs in order:
1. http://localhost:8501
2. http://127.0.0.1:8501
3. http://[WSL-IP]:8501 (shown when dashboard starts)

## 💡 Windows-Specific Solutions

### Option 1: Windows Terminal + Browser
```bash
# In Windows Terminal (WSL tab)
cd /home/republic/algotrade/algostack
source venv/bin/activate
streamlit run dashboard_app.py

# In Windows Browser
http://localhost:8501
```

### Option 2: Port Forwarding (PowerShell Admin)
```powershell
# Get your WSL IP first
wsl hostname -I

# Forward the port (replace with your IP)
netsh interface portproxy add v4tov4 listenport=8501 connectport=8501 connectaddress=[YOUR-WSL-IP]
```

### Option 3: Use WSL1 Instead of WSL2
WSL1 has better localhost support:
```powershell
wsl --set-version Ubuntu 1
```

## ❌ Common Issues

### "Module not found" errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### "Connection refused" errors
- Windows Firewall might be blocking
- Try disabling firewall temporarily
- Use a different port: `--server.port 8080`

### "Site can't be reached"
- Try all three URLs above
- Run the test server first
- Check if another app is using port 8501

## 🎯 Success Checklist

✅ Virtual environment activated  
✅ Streamlit installed  
✅ Dashboard starts without errors  
✅ Browser can connect to at least one URL  

## 🚀 Once It's Working

The dashboard will:
- Auto-discover all strategies
- Load configuration from config/base.yaml
- Allow backtesting with real data
- Show comprehensive metrics
- Export results to CSV/JSON

## Need More Help?

1. Check the terminal output for specific errors
2. Try the test server to verify networking
3. Review WSL_NETWORKING_GUIDE.md for advanced solutions
# WSL Dashboard Networking Guide

## Quick Solutions

### Option 1: Try Different URLs
When you run the dashboard, try these URLs in order:
1. `http://localhost:8501`
2. `http://127.0.0.1:8501`
3. `http://[WSL-IP]:8501` (shown when dashboard starts)

### Option 2: Use Alternative Launcher
```bash
cd algostack
python3 dashboard_alt.py
```

### Option 3: Port Forwarding (Run in Windows PowerShell as Admin)
```powershell
# Get WSL IP
wsl hostname -I

# Set up port forwarding (replace 192.168.x.x with your WSL IP)
netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=192.168.x.x

# To remove later:
netsh interface portproxy delete v4tov4 listenport=8501 listenaddress=0.0.0.0
```

## Common WSL Networking Issues

### 1. Windows Firewall Blocking
- Temporarily disable Windows Defender Firewall
- Or add an inbound rule for port 8501

### 2. WSL2 NAT Network
WSL2 uses a NAT network which can cause issues. Solutions:
- Use `localhost` forwarding (usually works)
- Set up port proxy (see Option 3)
- Use WSL1 instead of WSL2 for better localhost support

### 3. Browser Issues
- Use Chrome or Edge (not Internet Explorer)
- Clear browser cache
- Try incognito/private mode

## Diagnostic Commands

### Check WSL Version
```bash
wsl --list --verbose
```

### Check Network Configuration
```bash
# In WSL
ip addr show
netstat -tlnp | grep 8501
```

### Test Connectivity
```bash
# From Windows PowerShell
Test-NetConnection -ComputerName localhost -Port 8501
```

## Alternative: Run Dashboard on Windows Directly

If WSL networking continues to be problematic:

1. Install Python on Windows
2. Copy the algostack folder to Windows
3. Run directly in Windows:
   ```cmd
   cd C:\path\to\algostack
   pip install -r requirements.txt
   streamlit run dashboard_app.py
   ```

## Permanent Fix: WSL Configuration

Create or edit `~/.wslconfig` in Windows:
```ini
[wsl2]
localhostForwarding=true
```

Then restart WSL:
```powershell
wsl --shutdown
```

## Still Having Issues?

1. Check if another process is using port 8501:
   ```bash
   lsof -i :8501
   ```

2. Try a different port:
   ```bash
   streamlit run dashboard_app.py --server.port 8080
   ```

3. Check Streamlit logs:
   ```bash
   streamlit run dashboard_app.py --logger.level=debug
   ```

## Working Example

Here's what should work in most cases:

```bash
# Terminal 1 (WSL)
cd /home/republic/algotrade/algostack
python3 dashboard_alt.py

# Terminal 2 (Windows Browser)
# Navigate to http://localhost:8501
```

The dashboard will show multiple URLs - try each one until you find one that works!
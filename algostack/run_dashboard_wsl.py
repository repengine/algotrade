#!/usr/bin/env python3
"""
WSL-friendly dashboard runner
"""

import subprocess
import socket
import time
import sys
import os

def get_wsl_ip():
    """Get the WSL IP address"""
    result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
    if result.returncode == 0:
        ips = result.stdout.strip().split()
        return ips[0] if ips else 'localhost'
    return 'localhost'

def find_free_port(start=8501):
    """Find a free port"""
    for port in range(start, start + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except:
                continue
    return None

def main():
    # Activate virtual environment
    venv_python = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python3')
    
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found. Please run:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install streamlit pandas numpy plotly pyyaml yfinance")
        return 1
    
    # Find free port
    port = find_free_port()
    if not port:
        print("❌ No free ports available")
        return 1
    
    # Get WSL IP
    wsl_ip = get_wsl_ip()
    
    print("🚀 Starting AlgoStack Dashboard")
    print("=" * 50)
    print(f"Port: {port}")
    print(f"WSL IP: {wsl_ip}")
    print()
    print("Try these URLs in your Windows browser:")
    print(f"  • http://localhost:{port}")
    print(f"  • http://127.0.0.1:{port}")
    print(f"  • http://{wsl_ip}:{port}")
    print()
    print("If those don't work, run this in Windows PowerShell (Admin):")
    print(f"  netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 connectport={port} connectaddress={wsl_ip}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run streamlit
    cmd = [
        venv_python, '-m', 'streamlit', 'run',
        'dashboard_app_simple.py',
        '--server.port', str(port),
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
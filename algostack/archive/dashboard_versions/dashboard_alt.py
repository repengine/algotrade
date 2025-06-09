#!/usr/bin/env python3
"""
Alternative dashboard launcher with better WSL support
"""

import socket
import subprocess
import sys
from pathlib import Path


def get_local_ip():
    """Get local IP address"""
    try:
        # Create a socket and connect to an external address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def find_free_port(start_port=8501):
    """Find a free port starting from start_port"""
    port = start_port
    while port < start_port + 100:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    return None


def main():
    print("🚀 AlgoStack Dashboard Launcher (WSL Compatible)")
    print("=" * 50)

    # Get network info
    local_ip = get_local_ip()
    port = find_free_port()

    if not port:
        print("❌ Error: Could not find a free port")
        return 1

    print("\n📡 Network Configuration:")
    print(f"   Local IP: {local_ip}")
    print(f"   Port: {port}")

    print("\n🌐 Access URLs:")
    print(f"   • http://localhost:{port}")
    print(f"   • http://127.0.0.1:{port}")
    print(f"   • http://{local_ip}:{port}")

    print("\n⚠️  WSL Networking Tips:")
    print("   1. Try all three URLs above")
    print("   2. Disable Windows Firewall temporarily if needed")
    print("   3. Use Edge or Chrome (not Internet Explorer)")

    # Set environment variables for Streamlit
    import os

    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    # Change to app directory
    app_dir = Path(__file__).parent
    os.chdir(app_dir)

    print("\n🚀 Starting Streamlit dashboard...")
    print("   Press Ctrl+C to stop\n")

    # Run streamlit
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard_app.py",
                "--server.port",
                str(port),
                "--server.address",
                "0.0.0.0",
                "--server.headless",
                "true",
            ]
        )
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())

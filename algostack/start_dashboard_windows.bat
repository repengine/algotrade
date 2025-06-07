@echo off
echo Starting AlgoStack Dashboard for Windows/WSL...
echo.

REM Check if running from Windows or WSL
wsl --cd /home/republic/algotrade/algostack bash -c "streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"

echo.
echo Dashboard should open automatically in your browser.
echo If not, navigate to: http://localhost:8501
pause
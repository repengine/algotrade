#!/bin/bash
# Script to activate the virtual environment and set up the Python path

echo "Activating virtual environment..."
source venv/bin/activate

echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Test apscheduler import
python -c "from apscheduler.schedulers.asyncio import AsyncIOScheduler; print('✓ APScheduler import successful')"

echo ""
echo "Virtual environment activated!"
echo "You can now run Python scripts with all dependencies available."
echo ""
echo "To run a script, use: python <script_name.py>"
echo "To deactivate, type: deactivate"
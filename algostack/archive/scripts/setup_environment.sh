#!/bin/bash

# AlgoStack Environment Setup Script

echo "🚀 AlgoStack Environment Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

# Check for virtual environment support
echo -e "\n${YELLOW}Checking for virtual environment support...${NC}"

# Method 1: Try using existing venv if available
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo -e "${GREEN}✓ Existing virtual environment found${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
    
    # Upgrade pip
    echo -e "\n${YELLOW}Upgrading pip...${NC}"
    python -m pip install --upgrade pip
    
    # Install requirements
    echo -e "\n${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt
    
elif command_exists python3; then
    # Method 2: Try creating new venv
    echo -e "${YELLOW}Attempting to create virtual environment...${NC}"
    
    # Check if python3-venv is needed
    if ! python3 -m venv --help &>/dev/null; then
        echo -e "${RED}✗ Virtual environment module not available${NC}"
        echo -e "${YELLOW}To install it, run:${NC}"
        echo "    sudo apt-get update"
        echo "    sudo apt-get install python3-venv python3-pip"
        echo ""
        echo -e "${YELLOW}Alternative: Install packages with --break-system-packages flag:${NC}"
        echo "    pip install --break-system-packages -r requirements.txt"
        echo ""
        echo -e "${RED}Note: Using --break-system-packages is not recommended${NC}"
        exit 1
    fi
    
    # Create virtual environment
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Virtual environment created${NC}"
        source venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
        
        # Upgrade pip
        echo -e "\n${YELLOW}Upgrading pip...${NC}"
        python -m pip install --upgrade pip
        
        # Install requirements
        echo -e "\n${YELLOW}Installing requirements...${NC}"
        pip install -r requirements.txt
    else
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Check installation status
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Installation completed successfully!${NC}"
    echo -e "\n${YELLOW}To activate the virtual environment in the future, run:${NC}"
    echo "    source venv/bin/activate"
    echo -e "\n${YELLOW}To run the dashboard:${NC}"
    echo "    ./start_final_dashboard.sh"
else
    echo -e "\n${RED}❌ Installation failed${NC}"
    echo -e "${YELLOW}Common issues:${NC}"
    echo "1. Missing system packages - install with:"
    echo "   sudo apt-get install python3-venv python3-pip"
    echo ""
    echo "2. TA-Lib installation issues - install system library first:"
    echo "   sudo apt-get install ta-lib"
    echo ""
    echo "3. For a minimal installation without all dependencies:"
    echo "   pip install pandas numpy yfinance streamlit plotly"
fi
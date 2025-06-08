#!/bin/bash

# TA-Lib Installation Script for Ubuntu/Debian

echo "📊 TA-Lib Installation Helper"
echo "============================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Method 1: Install from source (most reliable)
echo -e "\n${YELLOW}Installing TA-Lib from source...${NC}"

# Install build dependencies
echo -e "${YELLOW}Installing build dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y wget build-essential

# Download TA-Lib
echo -e "\n${YELLOW}Downloading TA-Lib source...${NC}"
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

# Extract
echo -e "${YELLOW}Extracting...${NC}"
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

# Configure and build
echo -e "\n${YELLOW}Configuring and building TA-Lib...${NC}"
./configure --prefix=/usr
make

# Install
echo -e "${YELLOW}Installing TA-Lib...${NC}"
sudo make install

# Update library cache
sudo ldconfig

echo -e "\n${GREEN}✓ TA-Lib C library installed${NC}"

# Go back to project directory
cd /home/nate/projects/algotrade/algostack

# Now install Python wrapper
echo -e "\n${YELLOW}Installing TA-Lib Python wrapper...${NC}"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✓ Virtual environment detected${NC}"
    pip install TA-Lib
else
    echo -e "${YELLOW}No virtual environment active${NC}"
    echo -e "${YELLOW}Attempting to install with --user flag...${NC}"
    pip install --user TA-Lib || {
        echo -e "${RED}Failed to install with --user flag${NC}"
        echo -e "${YELLOW}You may need to use: pip install --break-system-packages TA-Lib${NC}"
    }
fi

echo -e "\n${GREEN}✅ TA-Lib installation complete!${NC}"
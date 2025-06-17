#!/bin/bash
# Wrapper script to run Python scripts with the virtual environment

if [ $# -eq 0 ]; then
    echo "Usage: ./run_with_venv.sh <python_script.py> [arguments]"
    exit 1
fi

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Use the virtual environment's Python
"$DIR/venv/bin/python" "$@"
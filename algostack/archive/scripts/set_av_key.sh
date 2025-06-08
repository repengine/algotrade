#!/bin/bash

# Extract Alpha Vantage API key from secrets.yaml and set as environment variable

# Check if secrets.yaml exists
if [ ! -f "config/secrets.yaml" ]; then
    echo "❌ Error: config/secrets.yaml not found"
    exit 1
fi

# Extract the API key using Python
API_KEY=$(python3 -c "
import yaml
with open('config/secrets.yaml', 'r') as f:
    secrets = yaml.safe_load(f)
    key = secrets.get('data_providers', {}).get('alphavantage', {}).get('api_key', '')
    print(key)
")

if [ -z "$API_KEY" ]; then
    echo "❌ No Alpha Vantage API key found in secrets.yaml"
else
    echo "✅ Alpha Vantage API key found"
    echo ""
    echo "To set it for this session, run:"
    echo "export ALPHA_VANTAGE_API_KEY='$API_KEY'"
    echo ""
    echo "Or source this script with:"
    echo "source set_av_key.sh"
    
    # If script is being sourced, export the variable
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        export ALPHA_VANTAGE_API_KEY="$API_KEY"
        echo "✅ API key exported to environment"
    fi
fi
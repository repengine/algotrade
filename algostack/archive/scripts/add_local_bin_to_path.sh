#!/bin/bash

# Add ~/.local/bin to PATH

echo "Adding ~/.local/bin to PATH..."

# Check which shell config file to use
if [ -f "$HOME/.bashrc" ]; then
    CONFIG_FILE="$HOME/.bashrc"
elif [ -f "$HOME/.zshrc" ]; then
    CONFIG_FILE="$HOME/.zshrc"
else
    CONFIG_FILE="$HOME/.profile"
fi

# Check if already in PATH
if echo $PATH | grep -q "$HOME/.local/bin"; then
    echo "✓ ~/.local/bin is already in PATH"
else
    # Add to config file
    echo '' >> $CONFIG_FILE
    echo '# Add local bin to PATH for pip installed packages' >> $CONFIG_FILE
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> $CONFIG_FILE
    
    echo "✓ Added to $CONFIG_FILE"
    echo ""
    echo "To apply changes:"
    echo "  source $CONFIG_FILE"
    echo ""
    echo "Or start a new terminal session"
fi

# Apply immediately for current session
export PATH="$HOME/.local/bin:$PATH"
echo "✓ PATH updated for current session"
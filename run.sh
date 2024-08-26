#!/bin/bash

# Define the virtual environment directory
VENV_DIR=".venv"

# Function to write colored output
function write_color {
    local message=$1
    local color=$2

    case $color in
        "yellow")
            echo -e "\033[1;33m${message}\033[0m"
            ;;
        "green")
            echo -e "\033[1;32m${message}\033[0m"
            ;;
        "cyan")
            echo -e "\033[1;36m${message}\033[0m"
            ;;
        *)
            echo "${message}"
            ;;
    esac
}

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    write_color "Creating virtual environment..." "yellow"
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
write_color "Activating virtual environment..." "green"
source "$VENV_DIR/bin/activate"

# Install dependencies
if [ -f "requirements.txt" ]; then
    write_color "Installing dependencies..." "yellow"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
fi

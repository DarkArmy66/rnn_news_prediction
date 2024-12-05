#!/bin/bash

# Function to detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     os=Linux;;
        Darwin*)    os=Mac;;
        CYGWIN*)    os=Cygwin;;
        MINGW*)     os=MinGw;;
        *)          os="UNKNOWN"
    esac
    echo $os
}

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook


#!/bin/bash
# PEARLNN Dependency Installation Script
# Installs all required dependencies for PEARLNN

set -e  # Exit on any error

echo "🔧 Installing PEARLNN Dependencies..."
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l) -eq 0 ]; then
    echo "❌ Python $REQUIRED_VERSION or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PEARLNN dependencies
echo "📦 Installing PEARLNN packages..."
pip install -r requirements.txt

# Install development dependencies if requested
if [ "$1" == "--dev" ]; then
    echo "🔧 Installing development dependencies..."
    pip install -e .[dev]
fi

# Install documentation dependencies if requested
if [ "$1" == "--docs" ]; then
    echo "📖 Installing documentation dependencies..."
    pip install -e .[docs]
fi

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Test installation: pearlnn --help"
echo "3. Run tests: pytest"
echo ""
echo "Happy parameter fitting! 🚀"
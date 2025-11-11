#!/bin/bash
# PEARLNN PyPI Upload Script
# Builds and uploads the package to PyPI

set -e  # Exit on any error

echo "🚀 PEARLNN PyPI Deployment"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Check if required tools are installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command -v twine &> /dev/null; then
    echo "❌ twine is required but not installed. Install with: pip install twine"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Run tests first
echo "🧪 Running tests..."
if ! python -m pytest pearlnn/tests/ -v; then
    echo "❌ Tests failed! Fix tests before deploying."
    exit 1
fi

# Check code quality
echo "📋 Checking code quality..."
if ! python -m black --check pearlnn/ scripts/; then
    echo "❌ Code formatting issues. Run 'black pearlnn/ scripts/' to fix."
    exit 1
fi

# Build the package
echo "📦 Building package..."
python -m build

# Check the built package
echo "🔍 Checking built package..."
twine check dist/*

# Ask for confirmation before uploading
echo ""
echo "📤 Ready to upload to PyPI"
echo "   The following files will be uploaded:"
ls -la dist/
echo ""

read -p "Are you sure you want to upload to PyPI? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Upload cancelled."
    exit 0
fi

# Upload to PyPI
echo "🚀 Uploading to PyPI..."
twine upload dist/*

echo ""
echo "✅ Upload completed successfully!"
echo "📦 Package is now available on PyPI"
echo "🔗 https://pypi.org/project/pearlnn/"
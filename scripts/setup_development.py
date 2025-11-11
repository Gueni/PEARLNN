#!/usr/bin/env python3
"""
PEARLNN Development Setup Script
Sets up development environment with pre-commit hooks and development tools
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        return False

def setup_development_environment():
    """Set up the complete development environment"""
    print("🚀 Setting up PEARLNN Development Environment")
    print("=" * 50)
    
    # Check if we're in the project root
    if not Path("pyproject.toml").exists():
        print("❌ Error: Please run this script from the project root directory")
        return False
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        return False
    
    # Set up pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        return False
    
    # Run pre-commit on all files
    if not run_command("pre-commit run --all-files", "Running pre-commit checks"):
        print("⚠️  Pre-commit checks failed, but continuing setup...")
    
    # Create necessary directories
    directories = [
        "data/models",
        "data/cache", 
        "data/user_data",
        "data/temp",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   📁 Created directory: {directory}")
    
    # Run tests to verify setup
    if not run_command("python -m pytest pearlnn/tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but setup completed")
    
    print("\n✅ Development environment setup completed!")
    print("\n🎯 Next steps:")
    print("   1. Activate virtual environment: source venv/bin/activate")
    print("   2. Make your changes")
    print("   3. Tests will run automatically on git commit")
    print("   4. Run 'python scripts/build_model.py' to build initial models")
    print("   5. Run 'python scripts/benchmark.py' to test performance")
    
    return True

if __name__ == "__main__":
    success = setup_development_environment()
    sys.exit(0 if success else 1)
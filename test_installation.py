#!/usr/bin/env python3

try:
    import pearlnn
    print("✅ PEARLNN package imported successfully!")
    
    # Test basic components
    from pearlnn.core.feature_extractor import WaveformFeatureExtractor
    from pearlnn.components.inductor import InductorAnalyzer
    from pearlnn.utils.config import Config
    
    print("✅ All modules imported successfully!")
    print(f"✅ PEARLNN version: {pearlnn.__version__}")
    
    # Test CLI
    from pearlnn.cli import create_parser
    parser = create_parser()
    print("✅ CLI parser created successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Troubleshooting steps:")
    print("1. Make sure you're in the PEARLNN directory")
    print("2. Run: pip install -e .")
    print("3. Check that all __init__.py files exist")
except Exception as e:
    print(f"❌ Error: {e}")
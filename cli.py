#!/usr/bin/env python3
"""
PEARLNN Command Line Interface
Parameter Extraction And Reverse Learning Neural Network
"""

#!/usr/bin/env python3
"""
PEARLNN Command Line Interface
Parameter Extraction And Reverse Learning Neural Network
"""

import argparse
import sys
import os
from pathlib import Path

# Fix imports - use relative imports
try:
    from .core.parameter_fitter import ParameterFitter
    from .core.feature_extractor import WaveformFeatureExtractor
    from .components.inductor import InductorAnalyzer
    from .community.model_sync import ModelSync
    from .utils.config import Config
    from .utils.logger import setup_logger
    from .utils.file_utils import FileManager
except ImportError:
    # Fallback for direct execution
    from core.parameter_fitter import ParameterFitter
    from core.feature_extractor import WaveformFeatureExtractor
    from components.inductor import InductorAnalyzer
    from community.model_sync import ModelSync
    from utils.config import Config
    from utils.logger import setup_logger
    from utils.file_utils import FileManager

import numpy as np
import torch

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger()
    
    try:
        if args.command == 'extract':
            handle_extract(args, logger)
        elif args.command == 'train':
            handle_train(args, logger)
        elif args.command == 'community':
            handle_community(args, logger)
        elif args.command == 'analyze':
            handle_analyze(args, logger)
        elif args.command == 'config':
            handle_config(args, logger)
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PEARLNN - Parameter Extraction And Reverse Learning Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pearlnn extract mosfet --csv waveform.csv
  pearlnn extract opamp --image response_curve.png --vcc 15
  pearlnn train mosfet --data training_data/ --epochs 2000
  pearlnn community sync
  pearlnn analyze capacitor --csv impedance.csv --frequency 1e6
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract parameters from data')
    extract_parser.add_argument('component', choices=['mosfet', 'bjt', 'opamp', 'capacitor', 'inductor', 'diode'],
                              help='Component type')
    extract_parser.add_argument('--csv', help='CSV file with waveform data')
    extract_parser.add_argument('--image', help='Image file with waveform plot')
    extract_parser.add_argument('--vcc', type=float, help='Supply voltage (V)')
    extract_parser.add_argument('--frequency', type=float, help='Test frequency (Hz)')
    extract_parser.add_argument('--output', '-o', help='Output file for results')
    extract_parser.add_argument('--uncertainty', action='store_true', help='Show uncertainty estimates')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model on new data')
    train_parser.add_argument('component', choices=['mosfet', 'bjt', 'opamp', 'capacitor', 'inductor', 'diode'],
                            help='Component type')
    train_parser.add_argument('--data', required=True, help='Directory with training data')
    train_parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Community command
    community_parser = subparsers.add_parser('community', help='Community features')
    community_parser.add_argument('action', choices=['sync', 'upload', 'stats', 'contributions'],
                                help='Community action')
    community_parser.add_argument('--component', help='Specific component')
    community_parser.add_argument('--model', help='Model file to upload')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze waveform characteristics')
    analyze_parser.add_argument('component', choices=['mosfet', 'bjt', 'opamp', 'capacitor', 'inductor', 'diode'],
                              help='Component type')
    analyze_parser.add_argument('--csv', help='CSV file with waveform data')
    analyze_parser.add_argument('--image', help='Image file with waveform plot')
    analyze_parser.add_argument('--features', action='store_true', help='Show extracted features')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('action', choices=['show', 'set', 'reset'], help='Config action')
    config_parser.add_argument('--key', help='Config key to set')
    config_parser.add_argument('--value', help='Config value to set')
    
    return parser

def handle_extract(args, logger):
    """Handle parameter extraction"""
    logger.info(f"Extracting {args.component} parameters...")
    
    # Initialize component analyzer
    if args.component == 'inductor':
        analyzer = InductorAnalyzer()
    else:
        analyzer = ParameterFitter(args.component)
    
    # Extract features from input data
    feature_extractor = WaveformFeatureExtractor()
    
    if args.csv:
        features = feature_extractor.extract_from_csv(args.csv)
    elif args.image:
        features = feature_extractor.extract_from_image(args.image)
    else:
        raise ValueError("Either --csv or --image must be provided")
    
    # Initialize parameter fitter
    fitter = ParameterFitter(args.component)
    
    # Try to load community model first
    try:
        community_sync = ModelSync()
        community_sync.download_latest_models([args.component])
        fitter.load_model(f"community/{args.component}_model.pt")
        logger.info("Loaded community model")
    except Exception as e:
        logger.warning(f"Could not load community model: {e}")
        logger.info("Using default model")
    
    # Extract parameters
    features_array = np.array(list(features.values())).reshape(1, -1)
    results = fitter.predict(features_array, n_samples=50 if args.uncertainty else 1)
    
    # Display results
    print(f"\n {args.component.upper()} Parameter Extraction Results:")
    print("=" * 50)
    
    for param_name, param_data in results.items():
        value = param_data['value']
        units = param_data.get('units', '')
        
        if args.uncertainty and 'uncertainty' in param_data:
            uncertainty = param_data['uncertainty']
            print(f"  {param_name:12}: {value:.6e} ± {uncertainty:.2e} {units}")
        else:
            print(f"  {param_name:12}: {value:.6e} {units}")
    
    # Save results if requested
    if args.output:
        file_mgr = FileManager()
        file_mgr.save_results(results, args.output)
        logger.info(f"Results saved to {args.output}")

def handle_train(args, logger):
    """Handle model training"""
    logger.info(f"Training {args.component} model...")
    
    # This would load training data and train the model
    # Implementation depends on data format and training procedure
    logger.info("Training feature coming soon!")
    
    # After training, optionally upload to community
    if hasattr(args, 'upload') and args.upload:
        community_sync = ModelSync()
        community_sync.upload_model("trained_model.pt", args.component)
        logger.info("Model uploaded to community")

def handle_community(args, logger):
    """Handle community operations"""
    community_sync = ModelSync()
    
    if args.action == 'sync':
        logger.info("Syncing with community...")
        downloaded = community_sync.download_latest_models()
        logger.info(f"Downloaded {len(downloaded)} models")
        
    elif args.action == 'upload':
        if not args.model:
            raise ValueError("--model must be specified for upload")
        if not args.component:
            raise ValueError("--component must be specified for upload")
        
        logger.info(f"Uploading {args.component} model to community...")
        results = community_sync.upload_model(args.model, args.component)
        logger.info("Upload completed")
        
    elif args.action == 'stats':
        stats = community_sync.get_community_stats()
        print("\n🌐 Community Statistics:")
        print("=" * 30)
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    elif args.action == 'contributions':
        contributions = community_sync.contribution_tracker.get_recent_contributions(10)
        print(f"\n📈 Recent Contributions ({len(contributions)}):")
        print("=" * 40)
        for contrib in contributions:
            print(f"  {contrib['timestamp']} - {contrib['component_type']}")

def handle_analyze(args, logger):
    """Handle waveform analysis"""
    logger.info(f"Analyzing {args.component} waveform...")
    
    feature_extractor = WaveformFeatureExtractor()
    
    if args.csv:
        features = feature_extractor.extract_from_csv(args.csv)
    elif args.image:
        features = feature_extractor.extract_from_image(args.image)
    else:
        raise ValueError("Either --csv or --image must be provided")
    
    if args.features:
        print(f"\n🔍 Extracted Features for {args.component}:")
        print("=" * 40)
        for feature_name, value in features.items():
            print(f"  {feature_name:25}: {value:.6e}")
    else:
        print(f"\n📈 Waveform Analysis Complete")
        print(f"  Extracted {len(features)} features")
        print(f"  Use --features to see detailed feature list")

def handle_config(args, logger):
    """Handle configuration management"""
    config = Config()
    
    if args.action == 'show':
        print("\n⚙️  PEARLNN Configuration:")
        print("=" * 30)
        for key, value in config.data.items():
            print(f"  {key}: {value}")
            
    elif args.action == 'set':
        if not args.key or not args.value:
            raise ValueError("Both --key and --value must be provided")
        
        # Convert value to appropriate type
        try:
            value = int(args.value)
        except ValueError:
            try:
                value = float(args.value)
            except ValueError:
                value = args.value
        
        config.set(args.key, value)
        config.save()
        logger.info(f"Set {args.key} = {value}")
        
    elif args.action == 'reset':
        config._load_defaults()
        config.save()
        logger.info("Configuration reset to defaults")

if __name__ == "__main__":
    main()
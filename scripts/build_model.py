#!/usr/bin/env python3
"""
PEARLNN Model Building Script
Builds and pre-trains initial models for various components
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add the parent directory to the path so we can import pearlnn
sys.path.insert(0, str(Path(__file__).parent.parent))

from pearlnn.core.bayesian_nn import BayesianElectronicsNN
from pearlnn.core.parameter_fitter import ParameterFitter
from pearlnn.utils.config import Config
from pearlnn.utils.file_utils import FileManager

def build_inductor_model():
    """Build and initialize the inductor parameter fitting model"""
    print("🔌 Building inductor model...")
    
    # Load configuration
    config = Config("config/default_config.yaml")
    inductor_config = Config("config/component_configs/inductor_config.yaml")
    
    # Get model parameters
    input_dim = inductor_config.get("model.input_features", 35)
    output_dim = len(inductor_config.get("parameters", []))
    hidden_dims = inductor_config.get("model.hidden_layers", [512, 256, 128])
    
    # Create model
    model = BayesianElectronicsNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims
    )
    
    # Initialize with sensible weights
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.1)
    
    model.apply(init_weights)
    
    # Create metadata
    metadata = {
        "component_type": "inductor",
        "model_architecture": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_layers": hidden_dims,
            "total_parameters": sum(p.numel() for p in model.parameters())
        },
        "training_status": "pretrained",
        "version": "1.0.0",
        "created_date": np.datetime64('now').astype(str),
        "parameters": inductor_config.get("parameters", []),
        "parameter_ranges": inductor_config.get("parameter_ranges", {})
    }
    
    # Save model
    file_mgr = FileManager()
    model_path = file_mgr.save_model(model, "inductor_pretrained.pt", metadata)
    
    print(f"✅ Inductor model built and saved to: {model_path}")
    print(f"   - Architecture: {input_dim} -> {hidden_dims} -> {output_dim}")
    print(f"   - Parameters: {metadata['model_architecture']['total_parameters']:,}")
    print(f"   - Output parameters: {', '.join(metadata['parameters'])}")
    
    return model_path

def create_dummy_training_data():
    """Create dummy training data for model testing"""
    print("📊 Creating dummy training data...")
    
    # Create synthetic inductor data for testing
    n_samples = 1000
    n_features = 35
    
    # Generate realistic feature ranges for inductors
    np.random.seed(42)  # For reproducible results
    
    features = np.random.normal(0, 1, (n_samples, n_features))
    
    # Generate realistic parameter values based on features
    # This is a simplified relationship for demonstration
    parameters = np.zeros((n_samples, 6))  # L, DCR, Q, SRF, Isat, ACR
    
    # Simulate some relationships
    parameters[:, 0] = 1e-6 * np.exp(0.5 * features[:, 0] + 0.3 * features[:, 1])  # L
    parameters[:, 1] = 0.1 * np.exp(0.2 * features[:, 2] + 0.1 * features[:, 3])   # DCR
    parameters[:, 2] = 50 + 20 * features[:, 4]                                    # Q_factor
    parameters[:, 3] = 10e6 * np.exp(0.4 * features[:, 5])                         # SRF
    parameters[:, 4] = 2.0 * np.exp(0.3 * features[:, 6])                          # Isat
    parameters[:, 5] = parameters[:, 1] * (1 + 0.1 * features[:, 7])               # ACR
    
    # Add some noise
    parameters += 0.1 * np.random.normal(0, 1, parameters.shape)
    
    # Ensure physical constraints
    parameters[:, 0] = np.abs(parameters[:, 0])  # L > 0
    parameters[:, 1] = np.abs(parameters[:, 1])  # DCR > 0
    parameters[:, 2] = np.maximum(parameters[:, 2], 1)  # Q >= 1
    parameters[:, 3] = np.maximum(parameters[:, 3], 1e6)  # SRF >= 1MHz
    parameters[:, 4] = np.abs(parameters[:, 4])  # Isat > 0
    parameters[:, 5] = np.maximum(parameters[:, 5], parameters[:, 1])  # ACR >= DCR
    
    # Save data
    file_mgr = FileManager()
    data = {
        "features": features.tolist(),
        "parameters": parameters.tolist(),
        "feature_names": [f"feature_{i}" for i in range(n_features)],
        "parameter_names": ["L", "DCR", "Q_factor", "SRF", "Isat", "ACR"],
        "description": "Synthetic inductor data for model testing",
        "n_samples": n_samples,
        "created_date": np.datetime64('now').astype(str)
    }
    
    data_path = file_mgr.save_results(data, "dummy_inductor_data.json")
    print(f"✅ Dummy training data saved to: {data_path}")
    
    return data_path

def validate_model(model_path):
    """Validate that the built model works correctly"""
    print("🔍 Validating model...")
    
    file_mgr = FileManager()
    model, metadata = file_mgr.load_model(model_path)
    
    # Test forward pass
    test_input = torch.randn(1, metadata["model_architecture"]["input_dim"])
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Model validation passed!")
    print(f"   - Input shape: {test_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    return True

def main():
    """Main function to build all models"""
    print("🏗️  PEARLNN Model Builder")
    print("=" * 50)
    
    try:
        # Build inductor model
        model_path = build_inductor_model()
        
        # Create dummy data for testing
        data_path = create_dummy_training_data()
        
        # Validate the model
        validate_model(model_path)
        
        print("\n🎉 All models built successfully!")
        print(f"📁 Model: {model_path}")
        print(f"📊 Data: {data_path}")
        
    except Exception as e:
        print(f"❌ Error building models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
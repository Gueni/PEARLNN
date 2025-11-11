import torch
import torch.nn as nn
import numpy as np
from .bayesian_nn import BayesianElectronicsNN
from .feature_extractor import WaveformFeatureExtractor

class ParameterFitter:
    """Main parameter fitting engine"""
    
    def __init__(self, component_type="mosfet"):
        self.component_type = component_type
        self.feature_extractor = WaveformFeatureExtractor()
        self.model = None
        self.is_trained = False
        
        # Component-specific parameter definitions
        self.parameter_configs = {
            "mosfet": {
                "parameters": ["Rds_on", "Ciss", "Coss", "Crss", "Qg", "Vth"],
                "input_dim": 50,  # Will be updated based on features
                "ranges": {
                    "Rds_on": (0.001, 10.0),
                    "Ciss": (1e-12, 1e-9),
                    "Coss": (1e-12, 1e-9),
                    "Crss": (1e-12, 1e-9),
                    "Qg": (1e-9, 1e-6),
                    "Vth": (0.5, 5.0)
                }
            },
            "opamp": {
                "parameters": ["GBW", "slew_rate", "Vos", "Ib", "CMRR"],
                "input_dim": 45,
                "ranges": {
                    "GBW": (1e3, 1e9),
                    "slew_rate": (0.1, 1000),
                    "Vos": (-0.01, 0.01),
                    "Ib": (1e-12, 1e-6),
                    "CMRR": (60, 140)
                }
            },
            "capacitor": {
                "parameters": ["ESR", "ESL", "leakage", "DF"],
                "input_dim": 35,
                "ranges": {
                    "ESR": (0.001, 10.0),
                    "ESL": (1e-12, 1e-6),
                    "leakage": (1e-12, 1e-3),
                    "DF": (0.001, 0.2)
                }
            }
        }
    
    def initialize_model(self, input_dim=None):
        """Initialize the Bayesian neural network"""
        if input_dim is None:
            input_dim = self.parameter_configs[self.component_type]["input_dim"]
        
        output_dim = len(self.parameter_configs[self.component_type]["parameters"])
        self.model = BayesianElectronicsNN(input_dim, output_dim)
        return self.model
    
    def fit(self, features, target_parameters, epochs=1000, learning_rate=0.001):
        """Train the model on features and target parameters"""
        if self.model is None:
            self.initialize_model(input_dim=features.shape[1])
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(target_parameters)
        
        # Normalize target parameters
        self._setup_normalization(y)
        y_normalized = self._normalize_targets(y)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Combined loss: MSE + KL divergence
        def loss_fn(predictions, targets, model):
            mse_loss = nn.MSELoss()(predictions, targets)
            kl_loss = model.kl_loss() / len(X)  # Normalize by batch size
            return mse_loss + 0.01 * kl_loss  # Weight KL loss
        
        # Training loop
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X, n_samples=1, sample=True)
            
            # Compute loss
            loss = loss_fn(predictions, y_normalized, self.model)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.is_trained = True
        return losses
    
    def predict(self, features, n_samples=50):
        """Predict parameters with uncertainty"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = torch.FloatTensor(features)
        
        # Get predictions with uncertainty
        mean_normalized, std_normalized = self.model.predict_with_uncertainty(X, n_samples=n_samples)
        
        # Denormalize predictions
        mean = self._denormalize_targets(mean_normalized)
        std = self._denormalize_targets(std_normalized)  # Approximate
        
        # Create result dictionary
        param_names = self.parameter_configs[self.component_type]["parameters"]
        results = {}
        
        for i, param_name in enumerate(param_names):
            results[param_name] = {
                "value": mean[0, i],
                "uncertainty": std[0, i],
                "units": self._get_units(param_name)
            }
        
        return results
    
    def _setup_normalization(self, targets):
        """Setup normalization for target parameters"""
        self.target_mean = torch.mean(targets, dim=0)
        self.target_std = torch.std(targets, dim=0)
        # Avoid division by zero
        self.target_std = torch.where(self.target_std == 0, torch.ones_like(self.target_std), self.target_std)
    
    def _normalize_targets(self, targets):
        """Normalize target parameters"""
        return (targets - self.target_mean) / self.target_std
    
    def _denormalize_targets(self, targets_normalized):
        """Denormalize target parameters"""
        return targets_normalized * self.target_std.numpy() + self.target_mean.numpy()
    
    def _get_units(self, parameter_name):
        """Get units for parameter"""
        units_map = {
            "Rds_on": "Ω", "Ciss": "F", "Coss": "F", "Crss": "F", 
            "Qg": "C", "Vth": "V", "GBW": "Hz", "slew_rate": "V/μs",
            "Vos": "V", "Ib": "A", "CMRR": "dB", "ESR": "Ω", 
            "ESL": "H", "leakage": "A", "DF": ""
        }
        return units_map.get(parameter_name, "")
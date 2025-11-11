import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters (mean and variance)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize means with small random values, variances with small positive values
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -6.0)  # ~0.01 variance initially
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -6.0)
    
    def forward(self, x, sample=True):
        if sample:
            # Sample weights and biases during training
            weight_epsilon = torch.randn_like(self.weight_rho)
            bias_epsilon = torch.randn_like(self.bias_rho)
            
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            
            weight = self.weight_mu + weight_epsilon * weight_sigma
            bias = self.bias_mu + bias_epsilon * bias_sigma
        else:
            # Use means during inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        """Compute KL divergence loss for this layer"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # KL divergence between learned distribution and prior N(0,1)
        kl_weight = -0.5 * torch.sum(1 + 2 * torch.log(weight_sigma) - self.weight_mu.pow(2) - weight_sigma.pow(2))
        kl_bias = -0.5 * torch.sum(1 + 2 * torch.log(bias_sigma) - self.bias_mu.pow(2) - bias_sigma.pow(2))
        
        return kl_weight + kl_bias

class BayesianElectronicsNN(nn.Module):
    """Bayesian Neural Network for electronics parameter fitting"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build Bayesian layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer (also Bayesian)
        layers.append(BayesianLinear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, n_samples=1, sample=True):
        """Forward pass with multiple samples for uncertainty estimation"""
        if n_samples > 1:
            # Monte Carlo sampling for uncertainty
            predictions = []
            for _ in range(n_samples):
                pred = self._forward_single(x, sample=True)
                predictions.append(pred)
            predictions = torch.stack(predictions)
            mean = predictions.mean(dim=0)
            std = predictions.std(dim-0)
            return mean, std
        else:
            return self._forward_single(x, sample)
    
    def _forward_single(self, x, sample=True):
        """Single forward pass"""
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                x = layer(x, sample=sample)
            else:
                x = layer(x)
        return x
    
    def kl_loss(self):
        """Total KL loss from all Bayesian layers"""
        total_kl = 0
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                total_kl += layer.kl_loss()
        return total_kl
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Predict with uncertainty quantification"""
        self.eval()
        with torch.no_grad():
            mean, std = self.forward(x, n_samples=n_samples, sample=True)
        return mean.numpy(), std.numpy()
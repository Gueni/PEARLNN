import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLoss(nn.Module):
    """Combined loss for Bayesian neural networks"""
    
    def __init__(self, kl_weight=0.01):
        super().__init__()
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, model):
        """Compute combined MSE + KL divergence loss"""
        mse = self.mse_loss(predictions, targets)
        kl = model.kl_loss() / len(targets)  # Normalize by batch size
        return mse + self.kl_weight * kl

class UncertaintyAwareLoss(nn.Module):
    """Loss function that accounts for prediction uncertainty"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mean_pred, std_pred, targets):
        """Negative log likelihood loss for uncertainty estimation"""
        # Assuming Gaussian distribution
        variance = std_pred ** 2 + 1e-8
        log_likelihood = -0.5 * torch.log(2 * torch.pi * variance) - 0.5 * (targets - mean_pred) ** 2 / variance
        return -torch.mean(log_likelihood)

class CompositeLoss(nn.Module):
    """Composite loss with multiple components"""
    
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = {'mse': 1.0, 'mae': 0.1, 'smoothness': 0.01}
        self.weights = weights
    
    def forward(self, predictions, targets, model=None):
        """Compute composite loss"""
        loss = 0
        
        if 'mse' in self.weights:
            mse = F.mse_loss(predictions, targets)
            loss += self.weights['mse'] * mse
        
        if 'mae' in self.weights:
            mae = F.l1_loss(predictions, targets)
            loss += self.weights['mae'] * mae
        
        if 'smoothness' in self.weights and predictions.shape[0] > 1:
            # Encourage smooth predictions
            smoothness = F.mse_loss(predictions[1:], predictions[:-1])
            loss += self.weights['smoothness'] * smoothness
        
        if model is not None and 'kl' in self.weights:
            kl = model.kl_loss() / len(targets)
            loss += self.weights['kl'] * kl
        
        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    """Bayesian linear layer with Gaussian variational inference"""
    
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -6.0)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -6.0)
    
    def forward(self, x, sample=True):
        """Forward pass with sampling"""
        if self.training or sample:
            # Sample from variational distribution
            weight = self.weight_mu + torch.randn_like(self.weight_rho) * F.softplus(self.weight_rho)
            bias = self.bias_mu + torch.randn_like(self.bias_rho) * F.softplus(self.bias_rho)
        else:
            # Use mean parameters
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        """Compute KL divergence with prior"""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # KL(q(w) || p(w)) where p(w) = N(0, prior_sigma^2 I)
        kl_weight = 0.5 * torch.sum(
            torch.log(torch.tensor(self.prior_sigma**2)) - 
            torch.log(weight_sigma**2) + 
            (weight_sigma**2 + self.weight_mu**2) / (self.prior_sigma**2) - 1
        )
        
        kl_bias = 0.5 * torch.sum(
            torch.log(torch.tensor(self.prior_sigma**2)) - 
            torch.log(bias_sigma**2) + 
            (bias_sigma**2 + self.bias_mu**2) / (self.prior_sigma**2) - 1
        )
        
        return kl_weight + kl_bias

class BayesianConv1d(nn.Module):
    """Bayesian 1D convolutional layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, prior_sigma=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma
        
        # Convolution parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -6.0)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -6.0)
    
    def forward(self, x, sample=True):
        """Forward pass with sampling"""
        if self.training or sample:
            weight = self.weight_mu + torch.randn_like(self.weight_rho) * F.softplus(self.weight_rho)
            bias = self.bias_mu + torch.randn_like(self.bias_rho) * F.softplus(self.bias_rho)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.conv1d(x, weight, bias, stride=self.stride, padding=self.padding)
    
    def kl_loss(self):
        """Compute KL divergence with prior"""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        kl_weight = 0.5 * torch.sum(
            torch.log(torch.tensor(self.prior_sigma**2)) - 
            torch.log(weight_sigma**2) + 
            (weight_sigma**2 + self.weight_mu**2) / (self.prior_sigma**2) - 1
        )
        
        kl_bias = 0.5 * torch.sum(
            torch.log(torch.tensor(self.prior_sigma**2)) - 
            torch.log(bias_sigma**2) + 
            (bias_sigma**2 + self.bias_mu**2) / (self.prior_sigma**2) - 1
        )
        
        return kl_weight + kl_bias
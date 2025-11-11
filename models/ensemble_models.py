import torch
import torch.nn as nn
import numpy as np

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, model_class, n_models=5, **model_kwargs):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([model_class(**model_kwargs) for _ in range(n_models)])
    
    def forward(self, x, n_samples=1):
        """Forward pass through ensemble"""
        if self.training or n_samples > 1:
            # Return all model outputs during training or uncertainty estimation
            outputs = [model(x, n_samples=1) for model in self.models]
            if isinstance(outputs[0], tuple):  # Bayesian models return (mean, std)
                means = torch.stack([out[0] for out in outputs])
                stds = torch.stack([out[1] for out in outputs])
                return means.mean(dim=0), stds.mean(dim=0)
            else:
                outputs = torch.stack(outputs)
                return outputs.mean(dim=0), outputs.std(dim=0)
        else:
            # Use mean prediction during inference
            outputs = [model(x, n_samples=1) for model in self.models]
            if isinstance(outputs[0], tuple):
                return torch.stack([out[0] for out in outputs]).mean(dim=0)
            else:
                return torch.stack(outputs).mean(dim=0)
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Predict with ensemble uncertainty"""
        self.eval()
        with torch.no_grad():
            all_predictions = []
            
            for model in self.models:
                if hasattr(model, 'predict_with_uncertainty'):
                    mean, std = model.predict_with_uncertainty(x, n_samples=n_samples//self.n_models)
                    # Sample from each model's predictive distribution
                    samples = np.random.normal(mean, std, (n_samples//self.n_models, *mean.shape))
                    all_predictions.append(samples)
                else:
                    predictions = model(torch.FloatTensor(x)).numpy()
                    all_predictions.append(predictions)
            
            all_predictions = np.concatenate(all_predictions, axis=0)
            mean = np.mean(all_predictions, axis=0)
            std = np.std(all_predictions, axis=0)
            
            return mean, std
    
    def train_ensemble(self, train_loader, val_loader=None, epochs=1000, **train_kwargs):
        """Train each model in the ensemble"""
        from ..training.trainer import Trainer
        
        losses = []
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}")
            trainer = Trainer(model)
            model_losses = trainer.train(train_loader, val_loader, epochs=epochs, **train_kwargs)
            losses.append(model_losses)
        
        return losses
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class Trainer:
    """Main training class for PEARLNN models"""
    
    def __init__(self, model, device='auto'):
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
    
    def _setup_device(self, device):
        """Setup training device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def train(self, train_loader, val_loader=None, epochs=1000, learning_rate=0.001, 
              patience=50, kl_weight=0.01):
        """Train the model with early stopping"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X, n_samples=1, sample=True)
                
                # Combined loss: MSE + KL divergence
                mse_loss = nn.MSELoss()(predictions, batch_y)
                kl_loss = self.model.kl_loss() / len(batch_X)
                loss = mse_loss + kl_weight * kl_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self.validate(val_loader, kl_weight)
                self.val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                scheduler.step(val_loss)
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 100 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")
        
        # Restore best model
        if val_loader is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return self.train_losses, self.val_losses if val_loader else None
    
    def validate(self, val_loader, kl_weight=0.01):
        """Validate model performance"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                predictions = self.model(batch_X, n_samples=1, sample=False)
                mse_loss = nn.MSELoss()(predictions, batch_y)
                kl_loss = self.model.kl_loss() / len(batch_X)
                loss = mse_loss + kl_weight * kl_loss
                
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def predict(self, test_loader, n_samples=50):
        """Make predictions with uncertainty"""
        self.model.eval()
        all_means = []
        all_stds = []
        
        with torch.no_grad():
            for batch_X in test_loader:
                batch_X = batch_X.to(self.device)
                mean, std = self.model.predict_with_uncertainty(batch_X, n_samples=n_samples)
                all_means.append(mean)
                all_stds.append(std)
        
        return np.concatenate(all_means), np.concatenate(all_stds)
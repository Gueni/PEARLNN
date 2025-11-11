import torch
import numpy as np
from collections import deque

class IncrementalLearner:
    """Handles incremental learning without catastrophic forgetting"""
    
    def __init__(self, model, memory_size=1000):
        self.model = model
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.importance_weights = {}
    
    def add_to_memory(self, features, targets):
        """Add new data to memory buffer"""
        for i in range(len(features)):
            if len(self.memory) >= self.memory_size:
                self.memory.popleft()
            self.memory.append((features[i], targets[i]))
    
    def compute_importance_weights(self, dataloader):
        """Compute importance weights for parameters"""
        self.model.eval()
        importance = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = torch.zeros_like(param)
        
        # Compute gradients on new data
        for batch_X, batch_y in dataloader:
            self.model.zero_grad()
            predictions = self.model(batch_X)
            loss = torch.nn.MSELoss()(predictions, batch_y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    importance[name] += param.grad.data ** 2
        
        # Normalize importance
        for name in importance:
            importance[name] /= len(dataloader)
        
        self.importance_weights = importance
        return importance
    
    def elastic_weight_consolidation(self, new_dataloader, importance_scale=1e6):
        """Apply Elastic Weight Consolidation to prevent forgetting"""
        
        # Compute importance weights on old data (from memory)
        if len(self.memory) > 0:
            memory_dataset = torch.utils.data.TensorDataset(
                torch.stack([torch.tensor(x) for x, _ in self.memory]),
                torch.stack([torch.tensor(y) for _, y in self.memory])
            )
            memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=32)
            self.compute_importance_weights(memory_loader)
        
        # EWC loss function
        def ewc_loss(new_data_loss):
            ewc_loss_val = 0
            for name, param in self.model.named_parameters():
                if name in self.importance_weights:
                    ewc_loss_val += (self.importance_weights[name] * (param - self.old_params[name]) ** 2).sum()
            return new_data_loss + (importance_scale / 2) * ewc_loss_val
        
        # Store old parameters
        self.old_params = {}
        for name, param in self.model.named_parameters():
            self.old_params[name] = param.data.clone()
        
        return ewc_loss
    
    def replay_learning(self, new_dataloader, epochs=100, replay_ratio=0.3):
        """Learn from new data while replaying old data"""
        
        if len(self.memory) == 0:
            # No memory yet, just train on new data
            return self._train_on_data(new_dataloader, epochs)
        
        # Combine new data with memory replay
        memory_dataset = torch.utils.data.TensorDataset(
            torch.stack([torch.tensor(x) for x, _ in self.memory]),
            torch.stack([torch.tensor(y) for _, y in self.memory])
        )
        
        # Sample from memory based on replay ratio
        replay_size = int(len(memory_dataset) * replay_ratio)
        if replay_size > 0:
            replay_subset = torch.utils.data.Subset(memory_dataset, 
                                                   np.random.choice(len(memory_dataset), replay_size, replace=False))
            combined_dataset = torch.utils.data.ConcatDataset([new_dataloader.dataset, replay_subset])
        else:
            combined_dataset = new_dataloader.dataset
        
        combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=new_dataloader.batch_size, shuffle=True)
        
        return self._train_on_data(combined_loader, epochs)
    
    def _train_on_data(self, dataloader, epochs):
        """Basic training on given data"""
        optimizer = torch.optim.Adam(self.model.parameters())
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = torch.nn.MSELoss()(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
        
        return losses
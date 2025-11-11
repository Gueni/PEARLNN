import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """Base model class with common functionality"""
    
    def __init__(self):
        super().__init__()
    
    def save(self, filepath):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config()
        }, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('model_config', {})
    
    def get_config(self):
        """Get model configuration"""
        return {
            'model_type': self.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def freeze_layers(self, layer_names=None):
        """Freeze specific layers or all layers"""
        if layer_names is None:
            # Freeze all layers
            for param in self.parameters():
                param.requires_grad = False
        else:
            # Freeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
    
    def unfreeze_layers(self, layer_names=None):
        """Unfreeze specific layers or all layers"""
        if layer_names is None:
            # Unfreeze all layers
            for param in self.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
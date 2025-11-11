import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration management for PEARLNN"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.data = {}
        
        if config_path and Path(config_path).exists():
            self.load(config_path)
        else:
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default configuration"""
        self.data = {
            'model': {
                'hidden_layers': [512, 256, 128],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 1000,
                'kl_weight': 0.01
            },
            'training': {
                'early_stopping_patience': 50,
                'validation_split': 0.2,
                'max_training_time': 3600  # 1 hour
            },
            'community': {
                'sync_on_startup': True,
                'auto_upload': True,
                'contribution_anonymity': True,
                'ipfs_gateways': [
                    'https://ipfs.io/ipfs/',
                    'https://gateway.pinata.cloud/ipfs/',
                    'https://cloudflare-ipfs.com/ipfs/'
                ]
            },
            'components': {
                'mosfet': {
                    'parameters': ['Rds_on', 'Ciss', 'Coss', 'Crss', 'Qg', 'Vth'],
                    'input_features': 50
                },
                'opamp': {
                    'parameters': ['GBW', 'slew_rate', 'Vos', 'Ib', 'CMRR'],
                    'input_features': 45
                },
                'capacitor': {
                    'parameters': ['ESR', 'ESL', 'leakage', 'DF'],
                    'input_features': 35
                }
            },
            'paths': {
                'models_dir': 'models',
                'data_dir': 'data',
                'cache_dir': 'cache',
                'community_dir': 'community'
            }
        }
    
    def load(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            self.data = yaml.safe_load(f)
        self.config_path = config_path
    
    def save(self, config_path=None):
        """Save configuration to YAML file"""
        if config_path is None:
            config_path = self.config_path
        
        if config_path:
            with open(config_path, 'w') as f:
                yaml.dump(self.data, f, indent=2, default_flow_style=False)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        current = self.data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        for key, value in updates.items():
            self.set(key, value)
    
    def get_component_config(self, component_type: str):
        """Get configuration for specific component"""
        return self.get(f'components.{component_type}', {})
    
    def get_model_config(self):
        """Get model configuration"""
        return self.get('model', {})
    
    def get_training_config(self):
        """Get training configuration"""
        return self.get('training', {})
    
    def get_community_config(self):
        """Get community configuration"""
        return self.get('community', {})
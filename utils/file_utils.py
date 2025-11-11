import os
import shutil
import json
import pickle
from pathlib import Path
from typing import Any, Dict

class FileManager:
    """File management utilities for PEARLNN"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            'models',
            'data/raw',
            'data/processed', 
            'data/cache',
            'results',
            'logs',
            'community'
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model, filename, metadata=None):
        """Save model with metadata"""
        model_path = self.base_dir / 'models' / filename
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': getattr(model, 'get_config', lambda: {})(),
            'metadata': metadata or {}
        }
        
        torch.save(save_data, model_path)
        return model_path
    
    def load_model(self, filename, model_class=None, model_kwargs=None):
        """Load model from file"""
        model_path = self.base_dir / 'models' / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if model_class is not None:
            model_config = checkpoint.get('model_config', {})
            if model_kwargs:
                model_config.update(model_kwargs)
            
            model = model_class(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, checkpoint.get('metadata', {})
        else:
            return checkpoint
    
    def save_results(self, results, filename):
        """Save results to JSON file"""
        results_path = self.base_dir / 'results' / filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        return results_path
    
    def load_results(self, filename):
        """Load results from JSON file"""
        results_path = self.base_dir / 'results' / filename
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def save_features(self, features, filename):
        """Save features to file"""
        features_path = self.base_dir / 'data/processed' / filename
        
        if filename.endswith('.pkl'):
            with open(features_path, 'wb') as f:
                pickle.dump(features, f)
        elif filename.endswith('.json'):
            with open(features_path, 'w') as f:
                json.dump(features, f, indent=2, default=self._json_serializer)
        elif filename.endswith('.npy'):
            np.save(features_path, features)
        else:
            raise ValueError("Unsupported file format")
        
        return features_path
    
    def load_features(self, filename):
        """Load features from file"""
        features_path = self.base_dir / 'data/processed' / filename
        
        if filename.endswith('.pkl'):
            with open(features_path, 'rb') as f:
                return pickle.load(f)
        elif filename.endswith('.json'):
            with open(features_path, 'r') as f:
                return json.load(f)
        elif filename.endswith('.npy'):
            return np.load(features_path)
        else:
            raise ValueError("Unsupported file format")
    
    def cache_data(self, data, key):
        """Cache data with key"""
        cache_path = self.base_dir / 'data/cache' / f"{key}.pkl"
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        return cache_path
    
    def get_cached_data(self, key, max_age_hours=24):
        """Get cached data if it exists and is fresh"""
        cache_path = self.base_dir / 'data/cache' / f"{key}.pkl"
        
        if not cache_path.exists():
            return None
        
        import time
        file_age = time.time() - cache_path.stat().st_mtime
        
        if file_age > max_age_hours * 3600:
            return None
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def list_models(self, pattern="*"):
        """List available models"""
        models_dir = self.base_dir / 'models'
        return list(models_dir.glob(pattern))
    
    def cleanup_old_files(self, directory, pattern="*", max_age_days=7):
        """Clean up old files"""
        target_dir = self.base_dir / directory
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for file_path in target_dir.glob(pattern):
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
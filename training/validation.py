import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelValidator:
    """Validate model performance and calibration"""
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_metrics(self, predictions, targets, uncertainties=None):
        """Compute comprehensive validation metrics"""
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # Relative errors
        relative_errors = np.abs((predictions - targets) / (targets + 1e-8))
        metrics['mean_relative_error'] = np.mean(relative_errors)
        metrics['median_relative_error'] = np.median(relative_errors)
        
        # Uncertainty calibration (if available)
        if uncertainties is not None:
            calibration_metrics = self._calibration_metrics(predictions, targets, uncertainties)
            metrics.update(calibration_metrics)
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calibration_metrics(self, predictions, targets, uncertainties):
        """Compute uncertainty calibration metrics"""
        metrics = {}
        
        # Normalized errors
        normalized_errors = np.abs(predictions - targets) / (uncertainties + 1e-8)
        metrics['mean_normalized_error'] = np.mean(normalized_errors)
        
        # Calibration: fraction of points within confidence intervals
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
        for confidence in confidence_levels:
            z_score = self._z_score_for_confidence(confidence)
            within_interval = np.sum(np.abs(predictions - targets) <= z_score * uncertainties) / len(targets)
            metrics[f'coverage_{int(confidence*100)}'] = within_interval
        
        # Sharpness (lower is better)
        metrics['mean_uncertainty'] = np.mean(uncertainties)
        metrics['uncertainty_std'] = np.std(uncertainties)
        
        return metrics
    
    def _z_score_for_confidence(self, confidence):
        """Get z-score for given confidence level"""
        from scipy import stats
        return stats.norm.ppf((1 + confidence) / 2)
    
    def cross_validate(self, model, features, targets, n_splits=5):
        """Perform k-fold cross-validation"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        for train_idx, val_idx in kf.split(features):
            # Split data
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]
            
            # Train model
            model.fit(X_train, y_train, epochs=100, verbose=0)
            
            # Predict and compute metrics
            predictions, uncertainties = model.predict(X_val)
            metrics = self.compute_metrics(predictions, y_val, uncertainties)
            fold_metrics.append(metrics)
        
        # Aggregate results
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            avg_metrics[key] = np.mean([fold[key] for fold in fold_metrics])
            avg_metrics[f'{key}_std'] = np.std([fold[key] for fold in fold_metrics])
        
        return avg_metrics, fold_metrics
    
    def learning_curve(self, model, features, targets, train_sizes=None):
        """Generate learning curve data"""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_scores = []
        val_scores = []
        
        for size in train_sizes:
            n_samples = int(size * len(features))
            
            # Split data
            X_train, X_val = features[:n_samples], features[n_samples:]
            y_train, y_val = targets[:n_samples], targets[n_samples:]
            
            # Train model
            model.fit(X_train, y_train, epochs=100, verbose=0)
            
            # Predict
            train_pred, _ = model.predict(X_train)
            val_pred, _ = model.predict(X_val)
            
            # Compute scores
            train_score = r2_score(y_train, train_pred)
            val_score = r2_score(y_val, val_pred)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        return {
            'train_sizes': train_sizes * len(features),
            'train_scores': train_scores,
            'val_scores': val_scores
        }
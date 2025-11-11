import numpy as np
import torch

class UncertaintyQuantifier:
    """Handles uncertainty quantification and calibration"""
    
    def __init__(self):
        self.calibration_data = []
    
    def calibrate_uncertainty(self, predictions, uncertainties, true_values):
        """Calibrate uncertainty estimates using validation data"""
        errors = np.abs(predictions - true_values)
        normalized_errors = errors / uncertainties
        
        # Store calibration data
        self.calibration_data.append((predictions, uncertainties, true_values))
        
        # Simple calibration: scale uncertainties based on empirical error distribution
        calibration_factor = np.median(normalized_errors)
        return calibration_factor
    
    def apply_calibration(self, uncertainties, calibration_factor):
        """Apply calibration to uncertainty estimates"""
        return uncertainties * calibration_factor
    
    def calculate_coverage(self, predictions, uncertainties, true_values, confidence=0.95):
        """Calculate coverage probability for given confidence interval"""
        z_score = self._z_score_for_confidence(confidence)
        lower_bounds = predictions - z_score * uncertainties
        upper_bounds = predictions + z_score * uncertainties
        
        coverage = np.mean((true_values >= lower_bounds) & (true_values <= upper_bounds))
        return coverage
    
    def _z_score_for_confidence(self, confidence):
        """Get z-score for given confidence level"""
        from scipy import stats
        return stats.norm.ppf((1 + confidence) / 2)
    
    def estimate_confidence_intervals(self, mean, std, confidence=0.95):
        """Calculate confidence intervals from mean and std"""
        z_score = self._z_score_for_confidence(confidence)
        lower = mean - z_score * std
        upper = mean + z_score * std
        return lower, upper
    
    def is_prediction_reliable(self, uncertainties, threshold=0.1):
        """Check if prediction is reliable based on uncertainty"""
        avg_uncertainty = np.mean(uncertainties)
        return avg_uncertainty < threshold
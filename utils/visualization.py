import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

class ResultVisualizer:
    """Visualization utilities for PEARLNN results"""
    
    def __init__(self, style='default'):
        self.set_style(style)
    
    def set_style(self, style='default'):
        """Set matplotlib style"""
        if style == 'default':
            plt.style.use('default')
            sns.set_palette("husl")
        elif style == 'seaborn':
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        elif style == 'dark':
            plt.style.use('dark_background')
    
    def plot_parameter_comparison(self, predicted, actual, parameter_names, uncertainties=None, figsize=(12, 8)):
        """Plot comparison between predicted and actual parameters"""
        n_params = len(parameter_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (param_name, ax) in enumerate(zip(parameter_names, axes)):
            pred_values = predicted[:, i]
            actual_values = actual[:, i]
            
            # Scatter plot
            ax.scatter(actual_values, pred_values, alpha=0.6, s=50)
            
            # Add uncertainty bars if available
            if uncertainties is not None:
                error = uncertainties[:, i]
                ax.errorbar(actual_values, pred_values, yerr=error, 
                           fmt='o', alpha=0.5, capsize=3)
            
            # Perfect prediction line
            min_val = min(actual_values.min(), pred_values.min())
            max_val = max(actual_values.max(), pred_values.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel(f'Actual {param_name}')
            ax.set_ylabel(f'Predicted {param_name}')
            ax.set_title(f'{param_name} Prediction')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self, train_losses, val_losses=None, figsize=(10, 6)):
        """Plot training history"""
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
        
        if val_losses is not None:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        return fig
    
    def plot_uncertainty_calibration(self, predictions, targets, uncertainties, figsize=(10, 6)):
        """Plot uncertainty calibration"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Calibration plot
        confidence_levels = np.linspace(0.1, 0.95, 10)
        empirical_coverage = []
        
        for confidence in confidence_levels:
            z_score = self._z_score_for_confidence(confidence)
            within_interval = np.mean(np.abs(predictions - targets) <= z_score * uncertainties)
            empirical_coverage.append(within_interval)
        
        ax1.plot(confidence_levels, empirical_coverage, 'bo-', label='Empirical')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Expected Coverage')
        ax1.set_ylabel('Empirical Coverage')
        ax1.set_title('Uncertainty Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error vs uncertainty plot
        normalized_errors = np.abs(predictions - targets) / (uncertainties + 1e-8)
        ax2.hist(normalized_errors, bins=30, alpha=0.7, density=True)
        ax2.axvline(1.0, color='r', linestyle='--', label='Ideal (mean=1)')
        ax2.set_xlabel('Normalized Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_waveform_comparison(self, original_time, original_values, 
                               simulated_time, simulated_values, figsize=(12, 6)):
        """Plot comparison between original and simulated waveforms"""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(original_time, original_values, 'b-', label='Original', alpha=0.8, linewidth=2)
        ax.plot(simulated_time, simulated_values, 'r--', label='Simulated', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def save_plot(self, fig, filename, dpi=300, bbox_inches='tight'):
        """Save plot to file"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
    
    def _z_score_for_confidence(self, confidence):
        """Get z-score for given confidence level"""
        from scipy import stats
        return stats.norm.ppf((1 + confidence) / 2)
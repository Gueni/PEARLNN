import numpy as np
import pandas as pd
import cv2
from scipy import signal
from scipy.fft import fft, fftfreq
import torch
from PIL import Image

class WaveformFeatureExtractor:
    """Extracts features from waveforms (CSV and images)"""
    
    def __init__(self):
        self.feature_names = []
    
    def extract_from_csv(self, csv_path, time_col=None, value_cols=None):
        """Extract features from CSV waveform data"""
        df = pd.read_csv(csv_path)
        
        if time_col is None:
            time_col = df.columns[0]
        if value_cols is None:
            value_cols = [col for col in df.columns if col != time_col]
        
        time = df[time_col].values
        features = {}
        
        for value_col in value_cols:
            values = df[value_col].values
            col_features = self._extract_waveform_features(time, values, prefix=value_col)
            features.update(col_features)
        
        return features
    
    def extract_from_image(self, image_path):
        """Extract waveform from datasheet image and get features"""
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Invert if needed (dark background, light waveform)
        if np.mean(image) > 127:
            image = cv2.bitwise_not(image)
        
        # Extract waveform curve
        time, values = self._extract_curve_from_image(image)
        
        # Extract features from the curve
        features = self._extract_waveform_features(time, values, prefix="image")
        return features
    
    def _extract_curve_from_image(self, image):
        """Extract waveform coordinates from image"""
        height, width = image.shape
        
        # Threshold to get binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find waveform using column-wise maxima
        time_axis = []
        value_axis = []
        
        for col in range(width):
            column = binary[:, col]
            if np.any(column > 0):
                # Find the top-most point (assuming waveform is white on dark background)
                y_positions = np.where(column > 0)[0]
                if len(y_positions) > 0:
                    y_pos = y_positions[0]  # Top of waveform
                    # Convert to normalized values (0-1)
                    normalized_y = 1.0 - (y_pos / height)
                    time_axis.append(col / width)  # Normalized time
                    value_axis.append(normalized_y)
        
        return np.array(time_axis), np.array(value_axis)
    
    def _extract_waveform_features(self, time, values, prefix=""):
        """Extract comprehensive features from waveform data"""
        features = {}
        
        if len(time) < 2:
            return features
        
        # Time domain features
        features[f'{prefix}_mean'] = np.mean(values)
        features[f'{prefix}_std'] = np.std(values)
        features[f'{prefix}_rms'] = np.sqrt(np.mean(values**2))
        features[f'{prefix}_peak_to_peak'] = np.ptp(values)
        features[f'{prefix}_skewness'] = self._skewness(values)
        features[f'{prefix}_kurtosis'] = self._kurtosis(values)
        features[f'{prefix}_crest_factor'] = np.max(np.abs(values)) / features[f'{prefix}_rms'] if features[f'{prefix}_rms'] != 0 else 0
        
        # Statistical features
        features[f'{prefix}_median'] = np.median(values)
        features[f'{prefix}_q25'] = np.percentile(values, 25)
        features[f'{prefix}_q75'] = np.percentile(values, 75)
        
        # Signal shape features
        features.update(self._extract_signal_shape_features(time, values, prefix))
        
        # Frequency domain features
        features.update(self._extract_frequency_features(time, values, prefix))
        
        self.feature_names = list(features.keys())
        return features
    
    def _extract_signal_shape_features(self, time, values, prefix):
        """Extract signal shape characteristics"""
        features = {}
        
        # Rise/fall times (for switching waveforms)
        if len(values) > 10:
            max_val = np.max(values)
            min_val = np.min(values)
            threshold_10 = min_val + 0.1 * (max_val - min_val)
            threshold_90 = min_val + 0.9 * (max_val - min_val)
            
            # Find crossing points
            rising_edges = np.where((values[:-1] < threshold_10) & (values[1:] >= threshold_10))[0]
            if len(rising_edges) > 0:
                rise_time = time[rising_edges[0] + 1] - time[rising_edges[0]]
                features[f'{prefix}_rise_time'] = rise_time
        
        # Overshoot
        if len(values) > 0:
            steady_state = np.mean(values[-len(values)//10:])  # Last 10%
            overshoot = (np.max(values) - steady_state) / steady_state if steady_state != 0 else 0
            features[f'{prefix}_overshoot'] = overshoot
        
        return features
    
    def _extract_frequency_features(self, time, values, prefix):
        """Extract frequency domain features"""
        features = {}
        
        if len(time) > 1:
            dt = time[1] - time[0] if len(time) > 1 else 1.0
            n = len(values)
            
            # FFT analysis
            fft_vals = fft(values - np.mean(values))  # Remove DC
            freqs = fftfreq(n, dt)
            
            # Only consider positive frequencies
            pos_freqs = freqs[:n//2]
            pos_magnitudes = np.abs(fft_vals[:n//2])
            
            if len(pos_magnitudes) > 0:
                # Dominant frequency
                dominant_idx = np.argmax(pos_magnitudes)
                features[f'{prefix}_dominant_freq'] = pos_freqs[dominant_idx]
                
                # Spectral energy
                features[f'{prefix}_spectral_energy'] = np.sum(pos_magnitudes**2)
                
                # Bandwidth (frequency where 90% of energy is contained)
                total_energy = np.sum(pos_magnitudes**2)
                cumulative_energy = np.cumsum(pos_magnitudes**2)
                if total_energy > 0:
                    bandwidth_idx = np.where(cumulative_energy >= 0.9 * total_energy)[0]
                    if len(bandwidth_idx) > 0:
                        features[f'{prefix}_bandwidth'] = pos_freqs[bandwidth_idx[0]]
        
        return features
    
    def _skewness(self, x):
        """Calculate skewness of data"""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.mean(((x - mean) / std) ** 3)
    
    def _kurtosis(self, x):
        """Calculate kurtosis of data"""
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.mean(((x - mean) / std) ** 4) - 3  # Excess kurtosis
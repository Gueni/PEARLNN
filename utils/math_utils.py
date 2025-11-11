import numpy as np
import scipy.stats as stats
from scipy import signal
from scipy.fft import fft, fftfreq

class MathUtils:
    """Mathematical utilities for signal processing and analysis"""
    
    @staticmethod
    def normalize_data(data, method='standardize', axis=0):
        """Normalize data using different methods"""
        if method == 'standardize':
            # Standardize to mean=0, std=1
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            return (data - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            # Scale to [0, 1] range
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            return (data - min_val) / (max_val - min_val + 1e-8)
        
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(data, axis=axis, keepdims=True)
            q75 = np.percentile(data, 75, axis=axis, keepdims=True)
            q25 = np.percentile(data, 25, axis=axis, keepdims=True)
            iqr = q75 - q25
            return (data - median) / (iqr + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def denormalize_data(normalized_data, original_data, method='standardize', axis=0):
        """Denormalize data"""
        if method == 'standardize':
            mean = np.mean(original_data, axis=axis, keepdims=True)
            std = np.std(original_data, axis=axis, keepdims=True)
            return normalized_data * std + mean
        
        elif method == 'minmax':
            min_val = np.min(original_data, axis=axis, keepdims=True)
            max_val = np.max(original_data, axis=axis, keepdims=True)
            return normalized_data * (max_val - min_val) + min_val
        
        elif method == 'robust':
            median = np.median(original_data, axis=axis, keepdims=True)
            q75 = np.percentile(original_data, 75, axis=axis, keepdims=True)
            q25 = np.percentile(original_data, 25, axis=axis, keepdims=True)
            iqr = q75 - q25
            return normalized_data * iqr + median
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def compute_correlation(x, y, method='pearson'):
        """Compute correlation between two signals"""
        if method == 'pearson':
            return np.corrcoef(x, y)[0, 1]
        elif method == 'spearman':
            return stats.spearmanr(x, y)[0]
        elif method == 'kendall':
            return stats.kendalltau(x, y)[0]
        else:
            raise ValueError(f"Unknown correlation method: {method}")
    
    @staticmethod
    def compute_snr(signal, noise=None):
        """Compute Signal-to-Noise Ratio"""
        if noise is None:
            # Assume noise is the difference from a smoothed version
            smoothed = MathUtils.smooth_signal(signal, window_size=5)
            noise = signal - smoothed
        
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def smooth_signal(signal, window_size=5, method='moving_average'):
        """Smooth signal using different methods"""
        if method == 'moving_average':
            window = np.ones(window_size) / window_size
            return np.convolve(signal, window, mode='same')
        
        elif method == 'savitzky_golay':
            return signal.savgol_filter(signal, window_size, 3)
        
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(signal, sigma=window_size/2)
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    @staticmethod
    def find_peaks(signal, height=None, distance=None, prominence=None):
        """Find peaks in signal"""
        peaks, properties = signal.find_peaks(signal, height=height, 
                                            distance=distance, prominence=prominence)
        return peaks, properties
    
    @staticmethod
    def compute_fft(signal, sampling_rate=1.0):
        """Compute FFT of signal"""
        n = len(signal)
        fft_result = fft(signal)
        freqs = fftfreq(n, 1/sampling_rate)
        
        # Return positive frequencies only
        pos_mask = freqs >= 0
        return freqs[pos_mask], np.abs(fft_result[pos_mask])
    
    @staticmethod
    def compute_entropy(signal, method='shannon'):
        """Compute entropy of signal"""
        if method == 'shannon':
            # Normalize to probability distribution
            hist, _ = np.histogram(signal, bins=50, density=True)
            hist = hist / np.sum(hist)
            return -np.sum(hist * np.log(hist + 1e-8))
        
        elif method == 'approximate':
            # Approximate entropy for time series
            return MathUtils._approximate_entropy(signal)
        
        else:
            raise ValueError(f"Unknown entropy method: {method}")
    
    @staticmethod
    def _approximate_entropy(signal, m=2, r=None):
        """Compute approximate entropy for time series"""
        if r is None:
            r = 0.2 * np.std(signal)
        
        def _phi(m):
            n = len(signal)
            patterns = np.lib.stride_tricks.sliding_window_view(signal, m)
            counts = np.zeros(len(patterns))
            
            for i, pattern in enumerate(patterns):
                distances = np.max(np.abs(patterns - pattern), axis=1)
                counts[i] = np.sum(distances <= r)
            
            return np.mean(np.log(counts / n))
        
        return _phi(m) - _phi(m + 1)
    
    @staticmethod
    def compute_complexity(signal):
        """Compute complexity measures for signal"""
        # Sample entropy (similar to approximate entropy)
        sampen = MathUtils._approximate_entropy(signal)
        
        # Lempel-Ziv complexity (normalized)
        lz_complexity = MathUtils._lempel_ziv_complexity(signal)
        
        return {
            'sample_entropy': sampen,
            'lempel_ziv_complexity': lz_complexity,
            'variance': np.var(signal),
            'kurtosis': stats.kurtosis(signal)
        }
    
    @staticmethod
    def _lempel_ziv_complexity(signal):
        """Compute Lempel-Ziv complexity"""
        # Convert to binary sequence based on median
        binary = (signal > np.median(signal)).astype(int)
        n = len(binary)
        
        i, k, l = 0, 1, 1
        c = 1
        
        while k + l <= n:
            if binary[i:i+l] == binary[k:k+l]:
                l += 1
            else:
                i += 1
                if i == k:
                    c += 1
                    k += l
                    l = 1
                    i = 0
                else:
                    l = 1
        
        # Normalize complexity
        return c * np.log2(n) / n
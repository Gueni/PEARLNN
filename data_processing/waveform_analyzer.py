import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

class WaveformAnalyzer:
    """Analyze waveform characteristics and extract features"""
    
    def __init__(self):
        self.features = {}
    
    def analyze_waveform(self, time, amplitude, waveform_type='auto'):
        """Comprehensive waveform analysis"""
        if waveform_type == 'auto':
            waveform_type = self._classify_waveform(time, amplitude)
        
        features = {
            'waveform_type': waveform_type,
            'basic_stats': self._basic_statistics(amplitude),
            'timing_features': self._timing_analysis(time, amplitude),
            'frequency_features': self._frequency_analysis(time, amplitude),
            'shape_features': self._shape_analysis(time, amplitude)
        }
        
        self.features = features
        return features
    
    def _classify_waveform(self, time, amplitude):
        """Classify waveform type"""
        # Simple classification based on characteristics
        if len(amplitude) < 10:
            return 'unknown'
        
        # Check for switching waveform
        diff = np.diff(amplitude)
        large_changes = np.sum(np.abs(diff) > 0.5 * np.ptp(amplitude))
        
        if large_changes >= 2:
            return 'switching'
        
        # Check for sinusoidal
        if self._is_sinusoidal(amplitude):
            return 'sinusoidal'
        
        # Check for pulse
        if self._is_pulse(amplitude):
            return 'pulse'
        
        return 'general'
    
    def _is_sinusoidal(self, amplitude):
        """Check if waveform is sinusoidal"""
        if len(amplitude) < 20:
            return False
        
        # Use FFT to check for dominant frequency
        fft_vals = np.abs(fft(amplitude - np.mean(amplitude)))
        fft_vals = fft_vals[:len(fft_vals)//2]
        
        # Count significant peaks
        peaks, _ = signal.find_peaks(fft_vals, height=0.1*np.max(fft_vals))
        return len(peaks) == 1  # Single dominant frequency
    
    def _is_pulse(self, amplitude):
        """Check if waveform is pulse-like"""
        # Look for rapid transitions between levels
        diff = np.diff(amplitude)
        large_transitions = np.sum(np.abs(diff) > 0.3 * np.ptp(amplitude))
        return large_transitions >= 2
    
    def _basic_statistics(self, amplitude):
        """Calculate basic statistical features"""
        return {
            'mean': np.mean(amplitude),
            'std': np.std(amplitude),
            'rms': np.sqrt(np.mean(amplitude**2)),
            'peak_to_peak': np.ptp(amplitude),
            'crest_factor': np.max(np.abs(amplitude)) / np.sqrt(np.mean(amplitude**2)) if np.mean(amplitude**2) > 0 else 0
        }
    
    def _timing_analysis(self, time, amplitude):
        """Analyze timing characteristics"""
        features = {}
        
        if len(time) < 2:
            return features
        
        # Rise and fall times (for switching waveforms)
        amp_range = np.ptp(amplitude)
        if amp_range > 0:
            threshold_10 = np.min(amplitude) + 0.1 * amp_range
            threshold_90 = np.min(amplitude) + 0.9 * amp_range
            
            # Find rise time (10% to 90%)
            rising_edges = self._find_crossings(amplitude, threshold_10, direction='rising')
            if len(rising_edges) > 0:
                rise_start = rising_edges[0]
                rise_end = self._find_crossings(amplitude[rise_start:], threshold_90, direction='rising')
                if len(rise_end) > 0:
                    features['rise_time'] = time[rise_start + rise_end[0]] - time[rise_start]
            
            # Find fall time (90% to 10%)
            falling_edges = self._find_crossings(amplitude, threshold_90, direction='falling')
            if len(falling_edges) > 0:
                fall_start = falling_edges[0]
                fall_end = self._find_crossings(amplitude[fall_start:], threshold_10, direction='falling')
                if len(fall_end) > 0:
                    features['fall_time'] = time[fall_start + fall_end[0]] - time[fall_start]
        
        return features
    
    def _frequency_analysis(self, time, amplitude):
        """Perform frequency domain analysis"""
        features = {}
        
        if len(time) < 2:
            return features
        
        dt = time[1] - time[0]
        n = len(amplitude)
        
        # FFT analysis
        fft_vals = fft(amplitude - np.mean(amplitude))
        freqs = fftfreq(n, dt)
        
        # Positive frequencies only
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitudes = np.abs(fft_vals[pos_mask])
        
        if len(pos_magnitudes) > 0:
            # Dominant frequency
            dominant_idx = np.argmax(pos_magnitudes)
            features['dominant_frequency'] = pos_freqs[dominant_idx]
            features['dominant_magnitude'] = pos_magnitudes[dominant_idx]
            
            # Bandwidth (frequency containing 90% of energy)
            total_energy = np.sum(pos_magnitudes**2)
            cumulative_energy = np.cumsum(pos_magnitudes**2)
            bandwidth_idx = np.where(cumulative_energy >= 0.9 * total_energy)[0]
            if len(bandwidth_idx) > 0:
                features['bandwidth'] = pos_freqs[bandwidth_idx[0]]
        
        return features
    
    def _shape_analysis(self, time, amplitude):
        """Analyze waveform shape characteristics"""
        features = {}
        
        # Overshoot and ringing
        if len(amplitude) > 10:
            steady_state = np.mean(amplitude[-len(amplitude)//10:])
            peak_value = np.max(amplitude)
            if steady_state != 0:
                features['overshoot'] = (peak_value - steady_state) / steady_state
        
        # Symmetry
        if len(amplitude) % 2 == 0:
            first_half = amplitude[:len(amplitude)//2]
            second_half = amplitude[len(amplitude)//2:]
            features['symmetry'] = np.corrcoef(first_half, second_half)[0,1] if len(first_half) > 1 else 0
        
        return features
    
    def _find_crossings(self, data, threshold, direction='rising'):
        """Find indices where data crosses threshold"""
        if direction == 'rising':
            crossings = np.where((data[:-1] < threshold) & (data[1:] >= threshold))[0]
        else:  # falling
            crossings = np.where((data[:-1] >= threshold) & (data[1:] < threshold))[0]
        return crossings
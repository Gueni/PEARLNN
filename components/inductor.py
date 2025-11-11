from .base_component import BaseComponent
import numpy as np

class InductorAnalyzer(BaseComponent):
    """Inductor parameter analyzer"""
    
    def __init__(self):
        super().__init__("inductor")
        self.parameters = ["L", "DCR", "Q_factor", "SRF", "Isat", "ACR"]
        self.parameter_ranges = {
            "L": (1e-9, 1.0),           # Inductance (H)
            "DCR": (0.001, 100.0),      # DC resistance (Ω)
            "Q_factor": (1, 500),       # Quality factor
            "SRF": (1e6, 10e9),         # Self-resonant frequency (Hz)
            "Isat": (0.001, 100.0),     # Saturation current (A)
            "ACR": (0.001, 10.0)        # AC resistance (Ω)
        }
        self.required_features = [
            "impedance_vs_frequency", "q_factor", "resonant_peak",
            "dc_resistance", "saturation_current"
        ]
    
    def extract_parameters(self, features):
        """Extract inductor parameters from waveform features"""
        params = {}
        
        # Estimate inductance from impedance characteristics
        if 'impedance_at_1mhz' in features and 'frequency' in features:
            # L = Z / (2πf) for ideal inductor
            impedance = features['impedance_at_1mhz']
            freq = 1e6  # 1 MHz reference
            params["L"] = impedance / (2 * np.pi * freq)
        else:
            params["L"] = 1e-6  # Default 1μH
        
        # DC resistance
        if 'dc_resistance' in features:
            params["DCR"] = features['dc_resistance']
        else:
            # Estimate from wire size and inductance
            params["DCR"] = 0.1 * np.sqrt(params["L"] * 1e6)  # Rough empirical
        
        # Quality factor
        if 'q_factor' in features:
            params["Q_factor"] = features['q_factor']
        else:
            # Estimate from inductance and frequency
            params["Q_factor"] = 100  # Typical value
        
        # Self-resonant frequency
        if 'resonant_frequency' in features:
            params["SRF"] = features['resonant_frequency']
        else:
            # Estimate from capacitance (typical parasitic)
            parasitic_c = 1e-12  # Typical parasitic capacitance
            params["SRF"] = 1 / (2 * np.pi * np.sqrt(params["L"] * parasitic_c))
        
        # Saturation current
        if 'saturation_current' in features:
            params["Isat"] = features['saturation_current']
        else:
            # Rough estimate based on size/inductance
            params["Isat"] = 1.0 / np.sqrt(params["L"])  # Empirical
        
        # AC resistance (skin effect + core losses)
        if 'ac_resistance' in features:
            params["ACR"] = features['ac_resistance']
        else:
            params["ACR"] = params["DCR"] * 1.1  # Slightly higher than DCR
        
        return params
    
    def validate_parameters(self, parameters):
        """Validate inductor parameters"""
        issues = []
        
        for param, value in parameters.items():
            if param in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param]
                if value < min_val or value > max_val:
                    issues.append(f"{param} value {value} outside typical range [{min_val}, {max_val}]")
        
        # Check consistency
        if "L" in parameters and "SRF" in parameters:
            # Higher inductance typically means lower SRF
            expected_srf = 1e9 / np.sqrt(parameters["L"] * 1e6)  # Rough empirical
            if abs(parameters["SRF"] - expected_srf) / expected_srf > 10:
                issues.append(f"SRF {parameters['SRF']:.1e} Hz seems inconsistent with L={parameters['L']:.1e} H")
        
        if "Q_factor" in parameters and "DCR" in parameters and "L" in parameters:
            # Q = ωL / R
            test_freq = 1e6  # 1 MHz test frequency
            calculated_q = (2 * np.pi * test_freq * parameters["L"]) / parameters["DCR"]
            if abs(parameters["Q_factor"] - calculated_q) / calculated_q > 5:
                issues.append(f"Q_factor inconsistent with L and DCR")
        
        return issues
    
    def estimate_core_type(self, parameters):
        """Estimate inductor core type based on parameters"""
        if "L" in parameters and "Isat" in parameters:
            inductance = parameters["L"]
            saturation_current = parameters["Isat"]
            
            # Simple classification based on energy storage capability
            energy = 0.5 * inductance * saturation_current**2
            
            if energy < 1e-6:
                return "Air Core"
            elif energy < 1e-3:
                return "Ferrite Core"
            else:
                return "Iron Powder Core"
        
        return "Unknown"
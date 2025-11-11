
from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """Base class for all electronic component analyzers"""
    
    def __init__(self, component_type):
        self.component_type = component_type
        self.parameters = []
        self.parameter_ranges = {}
        self.required_features = []
    
    @abstractmethod
    def extract_parameters(self, features):
        """Extract component parameters from features"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters):
        """Validate extracted parameters"""
        pass
    
    def get_parameter_info(self):
        """Get information about parameters for this component"""
        return {
            "component_type": self.component_type,
            "parameters": self.parameters,
            "ranges": self.parameter_ranges,
            "required_features": self.required_features
        }
    
    def normalize_parameters(self, parameters):
        """Normalize parameters to [0,1] range based on known ranges"""
        normalized = {}
        for param, value in parameters.items():
            if param in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param]
                normalized[param] = (value - min_val) / (max_val - min_val)
            else:
                normalized[param] = value
        return normalized
    
    def denormalize_parameters(self, normalized_parameters):
        """Denormalize parameters from [0,1] range to physical values"""
        denormalized = {}
        for param, value in normalized_parameters.items():
            if param in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param]
                denormalized[param] = value * (max_val - min_val) + min_val
            else:
                denormalized[param] = value
        return denormalized
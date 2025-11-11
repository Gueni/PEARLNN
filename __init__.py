"""PEARLNN - Parameter Extraction And Reverse Learning Neural Network"""

__version__ = "0.1.0"
__author__ = "PEARLNN Community"
__description__ = "AI-powered parameter fitting for electronics"

# Import key components to make them easily accessible
from .core.bayesian_nn import BayesianElectronicsNN
from .core.feature_extractor import WaveformFeatureExtractor
from .core.parameter_fitter import ParameterFitter
from .components.inductor import inductorAnalyzer
from .community.model_sync import ModelSync
from .utils.config import Config
from .utils.logger import setup_logger

__all__ = [
    'BayesianElectronicsNN',
    'WaveformFeatureExtractor', 
    'ParameterFitter',
    'inductorAnalyzer',
    'ModelSync',
    'Config',
    'setup_logger'
]
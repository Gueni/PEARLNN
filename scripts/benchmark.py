#!/usr/bin/env python3
"""
PEARLNN Benchmarking Script
Tests performance and accuracy of the parameter fitting system
"""

import time
import numpy as np
import torch
from pathlib import Path
import sys
import json

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pearlnn.core.parameter_fitter import ParameterFitter
from pearlnn.core.feature_extractor import WaveformFeatureExtractor
from pearlnn.components import InductorAnalyzer
from pearlnn.utils.config import Config
from pearlnn.utils.file_utils import FileManager

class Benchmark:
    """Benchmarking class for PEARLNN performance testing"""
    
    def __init__(self):
        self.config = Config()
        self.file_mgr = FileManager()
        self.results = {}
        
    def benchmark_feature_extraction(self, n_runs=10):
        """Benchmark feature extraction performance"""
        print("🔍 Benchmarking feature extraction...")
        
        feature_extractor = WaveformFeatureExtractor()
        
        # Create synthetic waveform data
        time = np.linspace(0, 1e-6, 1000)
        frequency = 1e6  # 1 MHz
        amplitude = np.sin(2 * np.pi * frequency * time) + 0.1 * np.random.normal(size=len(time))
        
        times = []
        for i in range(n_runs):
            start_time = time.time()
            
            # Extract features
            features = feature_extractor._extract_waveform_features(time, amplitude, "test")
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        stats = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "features_extracted": len(features),
            "runs": n_runs
        }
        
        self.results["feature_extraction"] = stats
        print(f"✅ Feature extraction: {stats['mean_time']*1000:.2f} ± {stats['std_time']*1000:.2f} ms per run")
        
        return stats
    
    def benchmark_model_inference(self, n_runs=50):
        """Benchmark model inference performance"""
        print("🧠 Benchmarking model inference...")
        
        # Initialize parameter fitter
        fitter = ParameterFitter("inductor")
        fitter.initialize_model(input_dim=35)
        
        # Create test features
        features = np.random.normal(0, 1, (1, 35))
        
        # Warm-up run
        _ = fitter.predict(features, n_samples=1)
        
        # Benchmark inference
        times_single = []
        times_uncertainty = []
        
        for i in range(n_runs):
            # Single sample (fast)
            start_time = time.time()
            result_single = fitter.predict(features, n_samples=1)
            end_time = time.time()
            times_single.append(end_time - start_time)
            
            # Uncertainty estimation (slower)
            start_time = time.time()
            result_uncertainty = fitter.predict(features, n_samples=50)
            end_time = time.time()
            times_uncertainty.append(end_time - start_time)
        
        stats = {
            "single_sample": {
                "mean_time": np.mean(times_single),
                "std_time": np.std(times_single),
                "min_time": np.min(times_single),
                "max_time": np.max(times_single),
            },
            "uncertainty_estimation": {
                "mean_time": np.mean(times_uncertainty),
                "std_time": np.std(times_uncertainty),
                "min_time": np.min(times_uncertainty),
                "max_time": np.max(times_uncertainty),
            },
            "runs": n_runs
        }
        
        self.results["model_inference"] = stats
        print(f"✅ Single sample: {stats['single_sample']['mean_time']*1000:.2f} ms")
        print(f"✅ Uncertainty (50 samples): {stats['uncertainty_estimation']['mean_time']*1000:.2f} ms")
        
        return stats
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage of different components"""
        print("💾 Benchmarking memory usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Memory before loading
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load feature extractor
        feature_extractor = WaveformFeatureExtractor()
        memory_after_fe = process.memory_info().rss / 1024 / 1024
        
        # Load parameter fitter
        fitter = ParameterFitter("inductor")
        fitter.initialize_model()
        memory_after_fitter = process.memory_info().rss / 1024 / 1024
        
        # Load component analyzer
        analyzer = InductorAnalyzer()
        memory_after_analyzer = process.memory_info().rss / 1024 / 1024
        
        stats = {
            "base_memory_mb": memory_before,
            "feature_extractor_mb": memory_after_fe - memory_before,
            "parameter_fitter_mb": memory_after_fitter - memory_after_fe,
            "component_analyzer_mb": memory_after_analyzer - memory_after_fitter,
            "total_memory_mb": memory_after_analyzer - memory_before
        }
        
        self.results["memory_usage"] = stats
        print(f"✅ Total memory usage: {stats['total_memory_mb']:.2f} MB")
        print(f"   - Feature extractor: {stats['feature_extractor_mb']:.2f} MB")
        print(f"   - Parameter fitter: {stats['parameter_fitter_mb']:.2f} MB")
        print(f"   - Component analyzer: {stats['component_analyzer_mb']:.2f} MB")
        
        return stats
    
    def benchmark_accuracy(self, n_tests=100):
        """Benchmark model accuracy on synthetic data"""
        print("🎯 Benchmarking accuracy...")
        
        # Load or create test data
        test_data_path = Path("data/user_data/benchmark_data.json")
        if test_data_path.exists():
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
            features = np.array(test_data["features"])
            true_params = np.array(test_data["parameters"])
        else:
            print("   Creating synthetic test data...")
            features = np.random.normal(0, 1, (n_tests, 35))
            # Simple synthetic relationship
            true_params = 0.5 * features[:, :6] + 0.1 * np.random.normal(0, 1, (n_tests, 6))
            
            # Save test data
            test_data = {
                "features": features.tolist(),
                "parameters": true_params.tolist(),
                "description": "Synthetic test data for accuracy benchmarking"
            }
            self.file_mgr.save_results(test_data, "benchmark_data.json")
        
        # Initialize fitter
        fitter = ParameterFitter("inductor")
        fitter.initialize_model(input_dim=35)
        
        # Make predictions
        predictions = []
        uncertainties = []
        
        for i in range(n_tests):
            result = fitter.predict(features[i:i+1], n_samples=50)
            pred_values = [result[param]["value"] for param in result.keys()]
            unc_values = [result[param]["uncertainty"] for param in result.keys()]
            
            predictions.append(pred_values)
            uncertainties.append(unc_values)
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        # Calculate accuracy metrics
        errors = np.abs(predictions - true_params)
        relative_errors = errors / (np.abs(true_params) + 1e-8)
        
        accuracy_stats = {
            "mean_absolute_error": np.mean(errors, axis=0).tolist(),
            "mean_relative_error": np.mean(relative_errors, axis=0).tolist(),
            "std_relative_error": np.std(relative_errors, axis=0).tolist(),
            "r_squared": self._calculate_r_squared(true_params, predictions),
            "parameter_names": ["L", "DCR", "Q_factor", "SRF", "Isat", "ACR"],
            "n_tests": n_tests
        }
        
        self.results["accuracy"] = accuracy_stats
        print(f"✅ Mean relative error: {np.mean(accuracy_stats['mean_relative_error']):.3f}")
        print(f"✅ R² score: {accuracy_stats['r_squared']:.3f}")
        
        return accuracy_stats
    
    def _calculate_r_squared(self, y_true, y_pred):
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("🏃 Running PEARLNN Benchmarks")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            self.benchmark_feature_extraction()
            self.benchmark_model_inference()
            self.benchmark_memory_usage()
            self.benchmark_accuracy()
            
            total_time = time.time() - start_time
            
            # Save results
            self.results["summary"] = {
                "total_benchmark_time": total_time,
                "timestamp": np.datetime64('now').astype(str),
                "pearlnn_version": "0.1.0"
            }
            
            # Save to file
            results_path = self.file_mgr.save_results(self.results, "benchmark_results.json")
            
            print("\n📊 Benchmark Summary")
            print("=" * 30)
            print(f"✅ All benchmarks completed in {total_time:.2f} seconds")
            print(f"📁 Results saved to: {results_path}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return None

def main():
    """Main benchmarking function"""
    benchmark = Benchmark()
    results = benchmark.run_all_benchmarks()
    
    if results:
        print("\n🎉 Benchmarks completed successfully!")
    else:
        print("\n💥 Benchmarks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
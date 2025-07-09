#!/usr/bin/env python3
"""
Simple Usage Examples for Guinier Analysis

This file demonstrates basic usage of the integrated GuinierAnalyzer
with both traditional and scikit-learn methods.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guinier_core import GuinierAnalyzer

def generate_sample_data():
    """Generate synthetic SAXS data for demonstration"""
    # Parameters for synthetic data
    Rg_true = 25.0  # Radius of gyration in Å
    I0_true = 1000.0  # Forward scattering intensity
    
    # Generate q values
    q = np.linspace(0.01, 0.15, 100)
    
    # Generate ideal Guinier curve
    I_ideal = I0_true * np.exp(-(q**2) * Rg_true**2 / 3)
    
    # Add some noise
    noise_level = 0.05
    I_noisy = I_ideal * (1 + noise_level * np.random.randn(len(q)))
    
    # Add some errors
    dI = I_noisy * noise_level
    
    # Ensure positive values
    I_noisy = np.maximum(I_noisy, 0.01)
    
    return q, I_noisy, dI

def basic_analysis_example():
    """Basic analysis workflow example"""
    print("=== Basic Analysis Example ===")
    
    # Initialize analyzer
    analyzer = GuinierAnalyzer()
    
    # Generate sample data
    q, I, dI = generate_sample_data()
    
    # Simulate loading data (in real usage, use analyzer.load_data(filename))
    analyzer.q_data = q
    analyzer.I_data = I
    analyzer.dI_data = dI
    analyzer.filtered_indices = np.arange(len(q))
    
    # Apply corrections
    result = analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=3.0)
    print(f"Corrections applied: {result['message']}")
    
    # Set automatic range
    range_result = analyzer.auto_range(q_rg_limit=1.3)
    print(f"Auto range: {range_result['message']}")
    
    # Perform traditional fit
    fit_result = analyzer.perform_fit(use_robust=True)
    if fit_result['success']:
        print(f"Traditional fit:")
        print(f"  Rg = {fit_result['Rg']:.2f} ± {fit_result['Rg_error']:.2f} Å")
        print(f"  I0 = {fit_result['I0']:.2e} ± {fit_result['I0_error']:.2e}")
        print(f"  R² = {fit_result['r_squared']:.4f}")
        print(f"  χ² = {fit_result['chi_squared']:.4f}")
        print(f"  Max q·Rg = {fit_result['max_q_rg']:.2f}")
    else:
        print(f"Traditional fit failed: {fit_result['message']}")
    
    print()

def sklearn_analysis_example():
    """Scikit-learn analysis example"""
    print("=== Scikit-learn Analysis Example ===")
    
    # Initialize analyzer
    analyzer = GuinierAnalyzer()
    
    # Generate sample data
    q, I, dI = generate_sample_data()
    
    # Simulate loading data
    analyzer.q_data = q
    analyzer.I_data = I
    analyzer.dI_data = dI
    analyzer.filtered_indices = np.arange(len(q))
    
    # Apply corrections and set range
    analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=3.0)
    analyzer.auto_range(q_rg_limit=1.3)
    
    if not analyzer.sklearn_available:
        print("Scikit-learn not available. Install with: pip install scikit-learn")
        return
    
    # Try different sklearn algorithms
    algorithms = ['linear', 'huber', 'ridge', 'theilsen']
    
    for algorithm in algorithms:
        print(f"\n--- {algorithm.upper()} Algorithm ---")
        result = analyzer.fit_with_sklearn(algorithm, cross_validate=True)
        
        if result['success']:
            print(f"  Rg = {result['Rg']:.2f} Å")
            print(f"  I0 = {result['I0']:.2e}")
            print(f"  R² = {result['r_squared']:.4f}")
            print(f"  χ² = {result['chi_squared']:.4f}")
            print(f"  Max q·Rg = {result['max_q_rg']:.2f}")
            
            if result['cv_mean'] is not None:
                print(f"  CV Score = {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
            
            if result['warning']:
                print(f"  WARNING: {result['warning']}")
        else:
            print(f"  Failed: {result['message']}")
    
    print()

def method_comparison_example():
    """Method comparison example"""
    print("=== Method Comparison Example ===")
    
    # Initialize analyzer
    analyzer = GuinierAnalyzer()
    
    # Generate sample data
    q, I, dI = generate_sample_data()
    
    # Simulate loading data
    analyzer.q_data = q
    analyzer.I_data = I
    analyzer.dI_data = dI
    analyzer.filtered_indices = np.arange(len(q))
    
    # Apply corrections and set range
    analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=3.0)
    analyzer.auto_range(q_rg_limit=1.3)
    
    # Compare all methods
    comparison = analyzer.compare_methods()
    
    if comparison:
        print("Method Comparison Results:")
        print("-" * 80)
        print(f"{'Method':<25} {'Rg (Å)':<10} {'I0':<12} {'R²':<8} {'χ²':<8} {'CV Score':<12}")
        print("-" * 80)
        
        for method_name, result in comparison.items():
            cv_score = f"{result.get('cv_mean', 0):.4f}" if result.get('cv_mean') is not None else "N/A"
            print(f"{result['method']:<25} {result['Rg']:<10.2f} {result['I0']:<12.2e} "
                  f"{result['r_squared']:<8.4f} {result['chi_squared']:<8.4f} {cv_score:<12}")
        
        print("-" * 80)
        
        # Get best sklearn model
        if analyzer.sklearn_available:
            best_model = analyzer.get_best_sklearn_model()
            if best_model['success']:
                print(f"\nRecommendation: {best_model['recommendation']}")
    else:
        print("Method comparison failed")
    
    print()

def file_usage_example():
    """Example of how to use with real files"""
    print("=== File Usage Example ===")
    print("This example shows how to use the analyzer with real data files:")
    print()
    
    code_example = '''
# Initialize analyzer
analyzer = GuinierAnalyzer()

# Load data from file
result = analyzer.load_data('your_data.grad')
if result['success']:
    print(f"Loaded {result['n_points']} data points")
    
    # Apply corrections
    analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=3.0)
    
    # Set automatic range
    analyzer.auto_range(q_rg_limit=1.3)
    
    # Perform fit
    fit_result = analyzer.perform_fit(use_robust=True)
    if fit_result['success']:
        print(f"Rg = {fit_result['Rg']:.2f} ± {fit_result['Rg_error']:.2f} Å")
        print(f"I0 = {fit_result['I0']:.2e} ± {fit_result['I0_error']:.2e}")
        
        # Save results
        analyzer.save_results('results.csv')
        print("Results saved to results.csv")
else:
    print(f"Failed to load data: {result['message']}")
'''
    
    print(code_example)

def main():
    """Run all examples"""
    print("Guinier Analysis Examples")
    print("=" * 50)
    print()
    
    # Check if scikit-learn is available
    try:
        import sklearn
        print("✓ Scikit-learn available")
    except ImportError:
        print("✗ Scikit-learn not available (install with: pip install scikit-learn)")
    
    print()
    
    # Run examples
    basic_analysis_example()
    sklearn_analysis_example()
    method_comparison_example()
    file_usage_example()
    
    print("Examples completed!")

if __name__ == "__main__":
    main() 
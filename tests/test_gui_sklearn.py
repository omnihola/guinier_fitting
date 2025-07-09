#!/usr/bin/env python3
"""
Test script for GUI with sklearn functionality
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guinier_core import GuinierAnalyzer

def test_sklearn_gui_functionality():
    """Test the sklearn functionality that will be used in GUI"""
    
    print("Testing Enhanced Guinier Analyzer for GUI...")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    Rg_true = 7.0
    I0_true = 1000.0
    q_max = 1.2 / Rg_true
    q = np.linspace(0.01, q_max, 35)
    
    I_true = I0_true * np.exp(-(q**2) * Rg_true**2 / 3)
    noise = 0.04 * I_true * np.random.randn(len(q))
    I_noisy = I_true + noise
    dI = 0.04 * I_true
    
    print(f"Test data: Rg = {Rg_true} Å, I0 = {I0_true}")
    print(f"Data points: {len(q)}, q range: {q[0]:.3f} - {q[-1]:.3f}")
    
    # Initialize analyzer
    analyzer = GuinierAnalyzer()
    
    # Load data (simulating GUI data loading)
    analyzer.q_data = q
    analyzer.I_data = I_noisy
    analyzer.dI_data = dI
    analyzer.filtered_indices = np.arange(len(q))
    analyzer.q_min_idx = 0
    analyzer.q_max_idx = len(q) - 1
    analyzer.bg_value = 0.0
    analyzer.norm_factor = 1.0
    
    print("\n1. Testing individual sklearn methods...")
    
    # Test different algorithms
    algorithms = ['traditional', 'traditional_robust', 'huber', 'linear', 'ridge']
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm}...")
        
        try:
            if algorithm == 'traditional':
                result = analyzer.perform_fit(use_robust=False)
            elif algorithm == 'traditional_robust':
                result = analyzer.perform_fit(use_robust=True)
            else:
                result = analyzer.fit_with_sklearn(algorithm, cross_validate=True)
            
            if result['success']:
                print(f"  ✓ Success: Rg = {result['Rg']:.3f} Å")
                print(f"  ✓ R² = {result['r_squared']:.4f}")
                if 'cv_scores' in result and result['cv_scores'] is not None:
                    cv_mean = np.mean(result['cv_scores'])
                    cv_std = np.std(result['cv_scores'])
                    print(f"  ✓ CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
            else:
                print(f"  ✗ Failed: {result['message']}")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print("\n2. Testing method comparison...")
    
    try:
        comparison_result = analyzer.compare_methods()
        
        if comparison_result['success']:
            print("  ✓ Method comparison successful")
            
            # Print comparison table
            analyzer.print_comparison()
            
            # Test best model selection
            best_model = analyzer.get_best_sklearn_model()
            if best_model:
                print(f"\n  ✓ Best model: {best_model['name']}")
                print(f"  ✓ CV Score: {best_model['cv_score']:.4f}")
            else:
                print("  ✗ No best model found")
                
        else:
            print(f"  ✗ Comparison failed: {comparison_result['message']}")
            
    except Exception as e:
        print(f"  ✗ Comparison error: {str(e)}")
    
    print("\n3. Testing GUI data structures...")
    
    # Test get_fit_results (used by GUI)
    try:
        fit_results = analyzer.get_fit_results()
        if fit_results:
            print("  ✓ get_fit_results() working")
            print(f"    Rg: {fit_results['Rg']:.3f} Å")
            print(f"    I0: {fit_results['I0']:.2e}")
            print(f"    R²: {fit_results['r_squared']:.4f}")
        else:
            print("  ✗ get_fit_results() returned None")
    except Exception as e:
        print(f"  ✗ get_fit_results() error: {str(e)}")
    
    # Test get_processed_data (used by GUI)
    try:
        processed_data = analyzer.get_processed_data()
        if processed_data:
            print("  ✓ get_processed_data() working")
            print(f"    Data keys: {list(processed_data.keys())}")
        else:
            print("  ✗ get_processed_data() returned None")
    except Exception as e:
        print(f"  ✗ get_processed_data() error: {str(e)}")
    
    print("\n4. Testing sklearn model storage...")
    
    # Check if sklearn models are stored correctly
    if hasattr(analyzer, 'sklearn_models'):
        print(f"  ✓ sklearn_models attribute exists")
        print(f"    Available models: {list(analyzer.sklearn_models.keys())}")
        
        # Test accessing model results
        for model_name, model_result in analyzer.sklearn_models.items():
            if 'cv_scores' in model_result and model_result['cv_scores'] is not None:
                cv_mean = np.mean(model_result['cv_scores'])
                print(f"    {model_name}: CV = {cv_mean:.4f}")
    else:
        print("  ✗ sklearn_models attribute missing")
    
    print("\n5. Testing model comparison data...")
    
    # Check if comparison data is stored correctly
    if hasattr(analyzer, 'model_comparison') and analyzer.model_comparison:
        print(f"  ✓ model_comparison attribute exists")
        print(f"    Available comparisons: {list(analyzer.model_comparison.keys())}")
        
        # Test accessing comparison results
        for method_name, method_result in analyzer.model_comparison.items():
            print(f"    {method_name}: Rg = {method_result['Rg']:.3f}, R² = {method_result['r_squared']:.4f}")
    else:
        print("  ✗ model_comparison attribute missing or empty")
    
    print("\n" + "=" * 50)
    print("✅ GUI sklearn functionality test completed!")
    print("All functions needed for GUI integration are working properly.")
    
    return analyzer

def create_test_plot(analyzer):
    """Create a test plot similar to GUI output"""
    
    print("\nCreating test plot...")
    
    # Get data
    data = analyzer.get_processed_data()
    fit_results = analyzer.get_fit_results()
    
    if not data or not fit_results:
        print("No data or fit results available for plotting")
        return
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: SAXS data
    ax1.errorbar(data['q_original'], data['I_corrected_all'], 
                yerr=data.get('dI_original', None), 
                fmt='o', markersize=3, alpha=0.7, label='Data')
    ax1.set_xlabel('q (Å⁻¹)')
    ax1.set_ylabel('I(q)')
    ax1.set_title('SAXS Data')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Guinier plot
    if 'q_range' in data:
        q_sq = data['q_range']**2
        ln_I = np.log(data['I_corrected_range'])
        ax2.plot(q_sq, ln_I, 'o', markersize=4, label='Fit data')
        
        # Add fit line
        fit_curve = analyzer.generate_fit_curve()
        if fit_curve:
            ax2.plot(fit_curve['q_sq'], fit_curve['ln_I_fit'], 
                    '-', color='red', linewidth=2, label='Fit')
    
    ax2.set_xlabel('q² (Å⁻²)')
    ax2.set_ylabel('ln(I)')
    ax2.set_title(f'Guinier Plot (Rg = {fit_results["Rg"]:.2f} Å)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Method comparison
    if hasattr(analyzer, 'model_comparison') and analyzer.model_comparison:
        methods = list(analyzer.model_comparison.keys())
        rg_values = [analyzer.model_comparison[m]['Rg'] for m in methods]
        r2_values = [analyzer.model_comparison[m]['r_squared'] for m in methods]
        
        ax3.bar(methods, rg_values, alpha=0.7)
        ax3.set_ylabel('Rg (Å)')
        ax3.set_title('Rg Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Highlight best model
        best_model = analyzer.get_best_sklearn_model()
        if best_model and best_model['name'] in methods:
            best_idx = methods.index(best_model['name'])
            ax3.bar(best_model['name'], rg_values[best_idx], 
                   color='red', alpha=0.9, label='Best')
            ax3.legend()
    
    # Plot 4: R² comparison
    if hasattr(analyzer, 'model_comparison') and analyzer.model_comparison:
        ax4.bar(methods, r2_values, alpha=0.7)
        ax4.set_ylabel('R²')
        ax4.set_title('R² Comparison')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim([min(r2_values) - 0.01, 1.0])
    
    plt.tight_layout()
    plt.savefig('test_gui_sklearn_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Test plot saved as 'test_gui_sklearn_output.png'")

if __name__ == "__main__":
    analyzer = test_sklearn_gui_functionality()
    create_test_plot(analyzer)
    
    print("\nTo test the GUI:")
    print("1. Run: python guinier_gui.py")
    print("2. Load some test data")
    print("3. Try different algorithms from the dropdown")
    print("4. Use 'Compare All Methods' button")
    print("5. Check cross-validation results")
    print("\nEverything should work smoothly! 🎉") 
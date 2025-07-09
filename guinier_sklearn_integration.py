#!/usr/bin/env python3
"""
Integration example: Adding scikit-learn capabilities to existing Guinier analysis

This module demonstrates how to extend the existing guinier_core.py module
with scikit-learn capabilities for enhanced fitting and model selection.
"""

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from guinier_core import GuinierAnalyzer
from guinier_sklearn import GuinierRegressor, SklearnGuinierAnalyzer


class EnhancedGuinierAnalyzer(GuinierAnalyzer):
    """
    Enhanced Guinier analyzer that combines the original functionality
    with scikit-learn capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.sklearn_models = {}
        self.model_comparison = None
        
    def fit_with_sklearn(self, regressor_name='huber', cross_validate=True):
        """
        Perform Guinier fitting using scikit-learn regressors.
        
        Parameters:
        -----------
        regressor_name : str
            Name of the sklearn regressor to use
        cross_validate : bool
            Whether to perform cross-validation
            
        Returns:
        --------
        dict : Enhanced fitting results
        """
        if self.q_data is None or self.I_data is None:
            return {'success': False, 'message': "No data loaded"}
        
        if self.filtered_indices is None:
            self.filtered_indices = np.arange(len(self.q_data))
        
        try:
            # Get filtered data
            q_filtered = self.q_data[self.filtered_indices]
            I_filtered = self.I_data[self.filtered_indices]
            
            # Get data in range for fitting
            q_range = q_filtered[self.q_min_idx:self.q_max_idx+1]
            I_range = I_filtered[self.q_min_idx:self.q_max_idx+1]
            
            # Apply corrections
            I_corrected = (I_range - self.bg_value) / self.norm_factor
            
            # Available regressors
            regressors = {
                'linear': LinearRegression(),
                'huber': HuberRegressor(epsilon=1.35, max_iter=100),
                'ridge': Ridge(alpha=1.0)
            }
            
            if regressor_name not in regressors:
                return {'success': False, 'message': f"Unknown regressor: {regressor_name}"}
            
            # Create Guinier regressor
            guinier_reg = GuinierRegressor(regressor=regressors[regressor_name])
            
            # Prepare sample weights if error data available
            sample_weights = None
            if self.dI_data is not None and self.filtered_indices is not None:
                dI_filtered = self.dI_data[self.filtered_indices]
                dI_range = dI_filtered[self.q_min_idx:self.q_max_idx+1] / self.norm_factor
                # For ln(I), error propagation gives σ_ln(I) = σ_I/I
                weights = I_corrected**2 / (dI_range**2)
                sample_weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
            
            # Fit the model
            guinier_reg.fit(q_range.reshape(-1, 1), I_corrected, sample_weight=sample_weights)
            
            # Store results in the original format
            self.Rg = guinier_reg.Rg_
            self.Rg_error = 0.0  # sklearn doesn't provide parameter errors directly
            self.I0 = guinier_reg.I0_
            self.I0_error = 0.0
            self.fit_slope = guinier_reg.slope_
            self.fit_intercept = guinier_reg.intercept_
            
            # Calculate fit quality metrics
            I_pred = guinier_reg.predict(q_range.reshape(-1, 1))
            ln_I = np.log(I_corrected)
            q_sq = q_range**2
            valid_idx = np.isfinite(ln_I)
            ln_I_pred = guinier_reg.regressor.predict(q_sq[valid_idx].reshape(-1, 1))
            
            self.r_squared = r2_score(ln_I[valid_idx], ln_I_pred)
            
            # Residuals for chi-squared
            residuals = ln_I[valid_idx] - ln_I_pred
            n = len(residuals)
            p = 2  # Two parameters
            if sample_weights is not None:
                chi_sq = np.sum(sample_weights[valid_idx] * residuals**2) / (n - p)
            else:
                chi_sq = np.sum(residuals**2) / (n - p)
            self.chi_squared = chi_sq
            
            # Cross-validation if requested
            cv_scores = None
            if cross_validate:
                try:
                    cv_scores = cross_val_score(
                        GuinierRegressor(regressor=regressors[regressor_name]),
                        q_range.reshape(-1, 1), I_corrected, cv=3, scoring='r2'
                    )
                except:
                    cv_scores = None
            
            # Physical validation
            max_q_rg = q_range[-1] * self.Rg
            warning = ""
            if max_q_rg > 1.3:
                warning = f"Warning: q·Rg exceeds 1.3 (max q·Rg = {max_q_rg:.2f})"
            
            result = {
                'success': True,
                'message': f"Sklearn {regressor_name} fit completed: Rg = {self.Rg:.2f} Å",
                'regressor': regressor_name,
                'Rg': self.Rg,
                'I0': self.I0,
                'r_squared': self.r_squared,
                'chi_squared': self.chi_squared,
                'max_q_rg': max_q_rg,
                'valid_guinier': max_q_rg <= 1.3,
                'cv_scores': cv_scores,
                'warning': warning,
                'sklearn_model': guinier_reg
            }
            
            # Store the sklearn model
            self.sklearn_models[regressor_name] = result
            
            return result
            
        except Exception as e:
            return {'success': False, 'message': f"Sklearn fitting failed: {str(e)}"}
    
    def compare_methods(self):
        """
        Compare traditional fitting with sklearn methods.
        
        Returns:
        --------
        dict : Comparison results
        """
        if self.q_data is None or self.I_data is None:
            return {'success': False, 'message': "No data loaded"}
        
        comparison = {}
        
        # Traditional method
        traditional_result = self.perform_fit(use_robust=False)
        if traditional_result['success']:
            comparison['traditional'] = {
                'method': 'numpy.polyfit',
                'Rg': traditional_result['Rg'],
                'I0': traditional_result['I0'],
                'r_squared': traditional_result['r_squared'],
                'chi_squared': traditional_result['chi_squared']
            }
        
        # Traditional robust method
        robust_result = self.perform_fit(use_robust=True)
        if robust_result['success']:
            comparison['traditional_robust'] = {
                'method': 'Theil-Sen/Huber',
                'Rg': robust_result['Rg'],
                'I0': robust_result['I0'],
                'r_squared': robust_result['r_squared'],
                'chi_squared': robust_result['chi_squared']
            }
        
        # Sklearn methods
        for method in ['linear', 'huber', 'ridge']:
            sklearn_result = self.fit_with_sklearn(method, cross_validate=True)
            if sklearn_result['success']:
                comparison[f'sklearn_{method}'] = {
                    'method': f'sklearn.{method}',
                    'Rg': sklearn_result['Rg'],
                    'I0': sklearn_result['I0'],
                    'r_squared': sklearn_result['r_squared'],
                    'chi_squared': sklearn_result['chi_squared'],
                    'cv_mean': np.mean(sklearn_result['cv_scores']) if sklearn_result['cv_scores'] is not None else None,
                    'cv_std': np.std(sklearn_result['cv_scores']) if sklearn_result['cv_scores'] is not None else None
                }
        
        self.model_comparison = comparison
        return {'success': True, 'comparison': comparison}
    
    def print_comparison(self):
        """Print a formatted comparison of different methods."""
        if self.model_comparison is None:
            print("No comparison available. Run compare_methods() first.")
            return
        
        print("\n" + "="*80)
        print("METHOD COMPARISON")
        print("="*80)
        print(f"{'Method':<20} {'Rg (Å)':<10} {'I0':<12} {'R²':<8} {'χ²':<8} {'CV_mean':<8} {'CV_std':<8}")
        print("-"*80)
        
        for name, result in self.model_comparison.items():
            cv_mean = f"{result.get('cv_mean', 0):.4f}" if result.get('cv_mean') is not None else "N/A"
            cv_std = f"{result.get('cv_std', 0):.4f}" if result.get('cv_std') is not None else "N/A"
            
            print(f"{result['method']:<20} "
                  f"{result['Rg']:<10.3f} "
                  f"{result['I0']:<12.2e} "
                  f"{result['r_squared']:<8.4f} "
                  f"{result['chi_squared']:<8.4f} "
                  f"{cv_mean:<8} "
                  f"{cv_std:<8}")
        
        print("="*80)
    
    def get_best_sklearn_model(self):
        """
        Get the best performing sklearn model based on cross-validation.
        
        Returns:
        --------
        dict : Best model information
        """
        if not self.sklearn_models:
            return None
        
        # Find model with highest CV score and valid Guinier regime
        best_model = None
        best_score = -np.inf
        
        for name, result in self.sklearn_models.items():
            if (result['valid_guinier'] and 
                result['cv_scores'] is not None and 
                np.mean(result['cv_scores']) > best_score):
                best_score = np.mean(result['cv_scores'])
                best_model = name
        
        if best_model:
            return {
                'name': best_model,
                'result': self.sklearn_models[best_model],
                'cv_score': best_score
            }
        
        return None


def demonstrate_integration():
    """
    Demonstrate the integration of sklearn with existing Guinier analysis.
    """
    print("Enhanced Guinier Analysis - sklearn Integration Demo")
    print("="*60)
    
    # Create enhanced analyzer
    analyzer = EnhancedGuinierAnalyzer()
    
    # Generate test data
    np.random.seed(123)
    Rg_true = 6.0
    I0_true = 800.0
    q_max = 1.2 / Rg_true  # Ensure valid Guinier regime
    q = np.linspace(0.02, q_max, 30)
    
    I_true = I0_true * np.exp(-(q**2) * Rg_true**2 / 3)
    noise = 0.03 * I_true * np.random.randn(len(q))
    I_noisy = I_true + noise
    dI = 0.03 * I_true
    
    print(f"Test data: Rg_true = {Rg_true} Å, I0_true = {I0_true}")
    print(f"Data points: {len(q)}, q range: {q[0]:.3f} - {q[-1]:.3f} Å⁻¹")
    
    # Load data using original interface
    analyzer.q_data = q
    analyzer.I_data = I_noisy
    analyzer.dI_data = dI
    analyzer.filtered_indices = np.arange(len(q))
    analyzer.q_min_idx = 0
    analyzer.q_max_idx = len(q) - 1
    
    # Compare methods
    print("\nComparing different fitting methods...")
    comparison_result = analyzer.compare_methods()
    
    if comparison_result['success']:
        analyzer.print_comparison()
        
        # Get best sklearn model
        best_sklearn = analyzer.get_best_sklearn_model()
        if best_sklearn:
            print(f"\nBest sklearn model: {best_sklearn['name']}")
            print(f"Cross-validation score: {best_sklearn['cv_score']:.4f}")
        
        # Show recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print("• Use 'linear' for clean data with few outliers")
        print("• Use 'huber' for data with potential outliers")
        print("• Use 'ridge' for noisy data (regularization helps)")
        print("• Traditional robust methods work well for comparison")
        print("• Cross-validation helps assess model stability")
        print("="*60)
    
    return analyzer


if __name__ == "__main__":
    # Run demonstration
    analyzer = demonstrate_integration()
    
    # Show how to use individual sklearn methods
    print("\n" + "="*60)
    print("INDIVIDUAL SKLEARN METHOD USAGE")
    print("="*60)
    
    # Example: Using specific sklearn method
    result = analyzer.fit_with_sklearn('huber', cross_validate=True)
    if result['success']:
        print(f"Huber regression result:")
        print(f"  Rg = {result['Rg']:.3f} Å")
        print(f"  R² = {result['r_squared']:.4f}")
        if result['cv_scores'] is not None:
            print(f"  CV scores: {result['cv_scores']}")
            print(f"  CV mean ± std: {np.mean(result['cv_scores']):.4f} ± {np.std(result['cv_scores']):.4f}")
    
    print("\n✅ Integration demonstration completed!")
    print("You can now use sklearn methods alongside traditional Guinier analysis!") 
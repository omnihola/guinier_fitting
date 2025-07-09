#!/usr/bin/env python3
"""
Guinier Analysis using scikit-learn

This module demonstrates how to use scikit-learn for Guinier fitting,
providing multiple regression algorithms and advanced features like
cross-validation, pipeline processing, and comprehensive model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    LinearRegression, HuberRegressor, RANSACRegressor, 
    TheilSenRegressor, Ridge, Lasso
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore')


class GuinierRegressor(BaseEstimator, RegressorMixin):
    """
    Custom scikit-learn compatible regressor for Guinier analysis.
    
    This wrapper transforms the Guinier equation into a linear regression problem
    and provides scikit-learn compatible interface.
    """
    
    def __init__(self, regressor=None, validate_range=True, max_qrg=1.3):
        """
        Initialize Guinier regressor.
        
        Parameters:
        -----------
        regressor : sklearn regressor
            Base regressor to use for fitting
        validate_range : bool
            Whether to validate q*Rg range
        max_qrg : float
            Maximum allowed q*Rg value
        """
        self.regressor = regressor if regressor else LinearRegression()
        self.validate_range = validate_range
        self.max_qrg = max_qrg
        
        # Fitted parameters
        self.Rg_ = None
        self.I0_ = None
        self.Rg_error_ = None
        self.I0_error_ = None
        self.slope_ = None
        self.intercept_ = None
        self.max_qrg_actual_ = None
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Guinier model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 1)
            q values
        y : array-like, shape (n_samples,)
            I values
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        self : object
        """
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else np.asarray(X)
        y = np.asarray(y)
        
        # Transform to linear problem: ln(I) vs q²
        q = X.flatten()
        q_sq = (q ** 2).reshape(-1, 1)
        ln_I = np.log(y)
        
        # Remove invalid points
        valid_mask = np.isfinite(ln_I) & (y > 0)
        q_sq_valid = q_sq[valid_mask]
        ln_I_valid = ln_I[valid_mask]
        
        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]
        
        # Fit the linear model
        if hasattr(self.regressor, 'fit'):
            if sample_weight is not None and hasattr(self.regressor, 'sample_weight'):
                self.regressor.fit(q_sq_valid, ln_I_valid, sample_weight=sample_weight)
            else:
                self.regressor.fit(q_sq_valid, ln_I_valid)
        
        # Extract parameters
        self.slope_ = self.regressor.coef_[0] if hasattr(self.regressor, 'coef_') else 0
        self.intercept_ = self.regressor.intercept_ if hasattr(self.regressor, 'intercept_') else 0
        
        # Calculate Guinier parameters
        if self.slope_ < 0:
            self.Rg_ = np.sqrt(-3 * self.slope_)
            self.I0_ = np.exp(self.intercept_)
            
            # Calculate maximum q*Rg
            self.max_qrg_actual_ = q[valid_mask][-1] * self.Rg_
            
            # Validate range if requested
            if self.validate_range and self.max_qrg_actual_ > self.max_qrg:
                warnings.warn(f"q*Rg = {self.max_qrg_actual_:.2f} exceeds {self.max_qrg}")
        else:
            raise ValueError("Negative slope required for valid Guinier fit")
        
        return self
    
    def predict(self, X):
        """
        Predict I values using the fitted Guinier model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 1)
            q values
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted I values
        """
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else np.asarray(X)
        q = X.flatten()
        
        if self.Rg_ is None or self.I0_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Guinier equation: I(q) = I0 * exp(-q²*Rg²/3)
        return self.I0_ * np.exp(-(q**2) * self.Rg_**2 / 3)
    
    def score(self, X, y, sample_weight=None):
        """
        Calculate R² score for the linear fit (ln(I) vs q²).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 1)
            q values
        y : array-like, shape (n_samples,)
            True I values
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights
            
        Returns:
        --------
        score : float
            R² score
        """
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else np.asarray(X)
        y = np.asarray(y)
        
        q = X.flatten()
        q_sq = (q ** 2).reshape(-1, 1)
        ln_I = np.log(y)
        
        valid_mask = np.isfinite(ln_I) & (y > 0)
        q_sq_valid = q_sq[valid_mask]
        ln_I_valid = ln_I[valid_mask]
        
        ln_I_pred = self.regressor.predict(q_sq_valid)
        
        return r2_score(ln_I_valid, ln_I_pred, sample_weight=sample_weight)


class SklearnGuinierAnalyzer:
    """
    Advanced Guinier analyzer using scikit-learn ecosystem.
    """
    
    def __init__(self):
        self.data = None
        self.fitted_models = {}
        self.best_model = None
        self.results = {}
        
    def load_data(self, q, I, dI=None):
        """
        Load SAXS data.
        
        Parameters:
        -----------
        q : array-like
            q values
        I : array-like
            Intensity values
        dI : array-like, optional
            Error values
        """
        self.data = {
            'q': np.asarray(q),
            'I': np.asarray(I),
            'dI': np.asarray(dI) if dI is not None else None
        }
        
    def get_regressors(self):
        """
        Get a dictionary of available regressors.
        
        Returns:
        --------
        dict : Dictionary of regressor name to regressor instance
        """
        return {
            'linear': LinearRegression(),
            'huber': HuberRegressor(epsilon=1.35, max_iter=100),
            'ransac': RANSACRegressor(random_state=42, max_trials=100, min_samples=0.5),
            'theilsen': TheilSenRegressor(random_state=42, max_subpopulation=1000),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1, max_iter=1000),
            # Skip problematic regressors for now
            # 'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            # 'svr': SVR(kernel='linear', C=1.0)
        }
    
    def create_pipeline(self, regressor, scaler=None):
        """
        Create a scikit-learn pipeline for Guinier fitting.
        
        Parameters:
        -----------
        regressor : sklearn regressor
            Base regressor
        scaler : sklearn scaler, optional
            Feature scaler
            
        Returns:
        --------
        Pipeline : scikit-learn pipeline
        """
        steps = []
        
        if scaler:
            steps.append(('scaler', scaler))
        
        steps.append(('guinier', GuinierRegressor(regressor=regressor)))
        
        return Pipeline(steps)
    
    def fit_multiple_models(self, q_range=None, use_weights=True, cv_folds=5):
        """
        Fit multiple regression models and compare their performance.
        
        Parameters:
        -----------
        q_range : tuple, optional
            (q_min, q_max) range for fitting
        use_weights : bool
            Whether to use error-based weights
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Results for each model
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        q, I, dI = self.data['q'], self.data['I'], self.data['dI']
        
        # Apply q range filter
        if q_range:
            mask = (q >= q_range[0]) & (q <= q_range[1])
            q, I = q[mask], I[mask]
            if dI is not None:
                dI = dI[mask]
        
        # Prepare sample weights
        sample_weights = None
        if use_weights and dI is not None:
            # Weight = 1/σ² for linear fit on ln(I)
            # Error propagation: σ_ln(I) = σ_I / I
            ln_I_errors = dI / I
            sample_weights = 1.0 / (ln_I_errors ** 2)
            sample_weights = np.where(np.isfinite(sample_weights), sample_weights, 1.0)
        
        regressors = self.get_regressors()
        results = {}
        
        for name, regressor in regressors.items():
            try:
                print(f"Fitting {name}...")
                
                # Create Guinier regressor
                guinier_reg = GuinierRegressor(regressor=regressor)
                
                # Fit the model
                guinier_reg.fit(q.reshape(-1, 1), I, sample_weight=sample_weights)
                
                # Calculate metrics
                I_pred = guinier_reg.predict(q.reshape(-1, 1))
                
                # Linear space metrics (ln(I) vs q²)
                q_sq = q ** 2
                ln_I = np.log(I)
                valid_mask = np.isfinite(ln_I)
                ln_I_pred = guinier_reg.regressor.predict(q_sq[valid_mask].reshape(-1, 1))
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(
                        GuinierRegressor(regressor=regressor.__class__(**regressor.get_params())),
                        q.reshape(-1, 1), I, cv=cv_folds, 
                        scoring='r2' if sample_weights is None else None
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except:
                    cv_mean, cv_std = np.nan, np.nan
                
                results[name] = {
                    'model': guinier_reg,
                    'Rg': guinier_reg.Rg_,
                    'I0': guinier_reg.I0_,
                    'max_qrg': guinier_reg.max_qrg_actual_,
                    'r2_linear': r2_score(ln_I[valid_mask], ln_I_pred),
                    'r2_original': r2_score(I, I_pred),
                    'rmse_linear': np.sqrt(mean_squared_error(ln_I[valid_mask], ln_I_pred)),
                    'rmse_original': np.sqrt(mean_squared_error(I, I_pred)),
                    'mae_linear': mean_absolute_error(ln_I[valid_mask], ln_I_pred),
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'valid_guinier': guinier_reg.max_qrg_actual_ <= 1.3
                }
                
                print(f"  Rg = {guinier_reg.Rg_:.2f} Å, R² = {results[name]['r2_linear']:.4f}")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.fitted_models = results
        
        # Select best model based on R² and Guinier validity
        valid_models = {k: v for k, v in results.items() 
                       if 'error' not in v and v.get('valid_guinier', False)}
        
        if valid_models:
            self.best_model = max(valid_models.keys(), 
                                key=lambda k: valid_models[k]['r2_linear'])
            print(f"\nBest model: {self.best_model}")
        else:
            print("\nNo valid models found!")
        
        return results
    
    def hyperparameter_tuning(self, regressor_name='huber', param_grid=None, cv_folds=5):
        """
        Perform hyperparameter tuning for a specific regressor.
        
        Parameters:
        -----------
        regressor_name : str
            Name of the regressor to tune
        param_grid : dict
            Parameter grid for GridSearchCV
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Tuning results
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        q, I = self.data['q'], self.data['I']
        
        # Default parameter grids
        default_param_grids = {
            'huber': {
                'guinier__regressor__epsilon': [1.1, 1.35, 1.5, 2.0],
                'guinier__regressor__alpha': [1e-4, 1e-3, 1e-2]
            },
            'ridge': {
                'guinier__regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'guinier__regressor__alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'random_forest': {
                'guinier__regressor__n_estimators': [50, 100, 200],
                'guinier__regressor__max_depth': [3, 5, 10, None]
            }
        }
        
        if param_grid is None:
            param_grid = default_param_grids.get(regressor_name, {})
        
        if not param_grid:
            raise ValueError(f"No parameter grid available for {regressor_name}")
        
        # Create pipeline
        regressors = self.get_regressors()
        if regressor_name not in regressors:
            raise ValueError(f"Unknown regressor: {regressor_name}")
        
        pipeline = self.create_pipeline(regressors[regressor_name])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv_folds, 
            scoring='r2', n_jobs=-1, verbose=1
        )
        
        print(f"Performing hyperparameter tuning for {regressor_name}...")
        grid_search.fit(q.reshape(-1, 1), I)
        
        best_model = grid_search.best_estimator_
        best_guinier = best_model.named_steps['guinier']
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': best_model,
            'Rg': best_guinier.Rg_,
            'I0': best_guinier.I0_,
            'max_qrg': best_guinier.max_qrg_actual_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best CV score: {results['best_score']:.4f}")
        print(f"Best Rg: {results['Rg']:.2f} Å")
        
        return results
    
    def plot_comparison(self, save_filename=None):
        """
        Plot comparison of different models.
        
        Parameters:
        -----------
        save_filename : str, optional
            Filename to save the plot
        """
        if not self.fitted_models:
            raise ValueError("Models must be fitted first")
        
        q, I = self.data['q'], self.data['I']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original data with fits
        ax1.scatter(q, I, alpha=0.6, s=20, label='Data')
        ax1.set_xlabel('q (Å⁻¹)')
        ax1.set_ylabel('I(q)')
        ax1.set_title('SAXS Data with Fits')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # Plot 2: Guinier plot
        q_sq = q ** 2
        ln_I = np.log(I)
        valid_mask = np.isfinite(ln_I)
        
        ax2.scatter(q_sq[valid_mask], ln_I[valid_mask], alpha=0.6, s=20, label='Data')
        ax2.set_xlabel('q² (Å⁻²)')
        ax2.set_ylabel('ln(I)')
        ax2.set_title('Guinier Plot with Fits')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.fitted_models)))
        
        for i, (name, result) in enumerate(self.fitted_models.items()):
            if 'error' in result:
                continue
                
            model = result['model']
            color = colors[i]
            
            # Original space fit
            I_pred = model.predict(q.reshape(-1, 1))
            ax1.plot(q, I_pred, '--', color=color, alpha=0.8, 
                    label=f'{name} (Rg={result["Rg"]:.2f})')
            
            # Guinier space fit
            ln_I_pred = model.regressor.predict(q_sq.reshape(-1, 1))
            ax2.plot(q_sq, ln_I_pred, '--', color=color, alpha=0.8, 
                    label=f'{name} (R²={result["r2_linear"]:.3f})')
        
        ax1.legend()
        ax2.legend()
        
        # Plot 3: R² comparison
        valid_results = {k: v for k, v in self.fitted_models.items() if 'error' not in v}
        names = list(valid_results.keys())
        r2_scores = [valid_results[name]['r2_linear'] for name in names]
        
        bars = ax3.bar(names, r2_scores, alpha=0.7)
        ax3.set_ylabel('R² Score')
        ax3.set_title('Model Comparison (R²)')
        ax3.set_ylim([min(r2_scores) - 0.01, 1.0])
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Highlight best model
        if self.best_model and self.best_model in names:
            best_idx = names.index(self.best_model)
            bars[best_idx].set_color('red')
            bars[best_idx].set_alpha(1.0)
        
        # Plot 4: Rg comparison
        rg_values = [valid_results[name]['Rg'] for name in names]
        valid_guinier = [valid_results[name]['valid_guinier'] for name in names]
        
        bars = ax4.bar(names, rg_values, alpha=0.7)
        ax4.set_ylabel('Rg (Å)')
        ax4.set_title('Rg Comparison')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Color code by validity
        for i, (bar, valid) in enumerate(zip(bars, valid_guinier)):
            if valid:
                bar.set_color('green')
            else:
                bar.set_color('orange')
        
        # Add legend for validity
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Valid (q·Rg ≤ 1.3)'),
                          Patch(facecolor='orange', label='Invalid (q·Rg > 1.3)')]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_filename}")
        
        plt.show()
    
    def get_summary_table(self):
        """
        Get a summary table of all fitted models.
        
        Returns:
        --------
        DataFrame : Summary table
        """
        if not self.fitted_models:
            raise ValueError("Models must be fitted first")
        
        data = []
        for name, result in self.fitted_models.items():
            if 'error' in result:
                data.append({
                    'Model': name,
                    'Status': 'Failed',
                    'Rg (Å)': 'N/A',
                    'I0': 'N/A',
                    'R²_linear': 'N/A',
                    'max_q·Rg': 'N/A',
                    'Valid_Guinier': 'N/A',
                    'Error': result['error'][:50] + '...' if len(result['error']) > 50 else result['error']
                })
            else:
                data.append({
                    'Model': name,
                    'Status': 'Success',
                    'Rg (Å)': f"{result['Rg']:.3f}",
                    'I0': f"{result['I0']:.2e}",
                    'R²_linear': f"{result['r2_linear']:.4f}",
                    'max_q·Rg': f"{result['max_qrg']:.3f}",
                    'Valid_Guinier': 'Yes' if result['valid_guinier'] else 'No',
                    'Error': 'None'
                })
        
        df = pd.DataFrame(data)
        return df


def demo_sklearn_guinier():
    """
    Demonstration of sklearn-based Guinier analysis.
    """
    print("Scikit-learn Guinier Analysis Demo")
    print("=" * 40)
    
    # Generate synthetic data with valid Guinier range
    np.random.seed(42)
    # Use smaller q range to ensure q*Rg <= 1.3
    Rg_true = 8.0  # True Rg
    q_max_valid = 1.3 / Rg_true  # Maximum q for valid Guinier regime ≈ 0.16
    q = np.linspace(0.01, q_max_valid * 0.9, 40)  # Slightly below limit
    I0_true = 1000.0  # True I0
    
    # Guinier curve with noise
    I_true = I0_true * np.exp(-(q**2) * Rg_true**2 / 3)
    noise_level = 0.05
    I_noise = I_true * (1 + noise_level * np.random.randn(len(q)))
    dI = noise_level * I_true
    
    # Add some outliers
    outlier_indices = np.random.choice(len(q), 3, replace=False)
    I_noise[outlier_indices] *= np.random.uniform(0.7, 1.5, 3)
    
    print(f"True parameters: Rg = {Rg_true} Å, I0 = {I0_true}")
    print(f"Data points: {len(q)}, Noise level: {noise_level*100}%")
    
    # Initialize analyzer
    analyzer = SklearnGuinierAnalyzer()
    analyzer.load_data(q, I_noise, dI)
    
    # Fit multiple models
    print("\nFitting multiple models...")
    results = analyzer.fit_multiple_models(use_weights=True, cv_folds=3)
    
    # Display summary
    print("\nModel Summary:")
    summary_df = analyzer.get_summary_table()
    print(summary_df.to_string(index=False))
    
    # Hyperparameter tuning for best model
    if analyzer.best_model:
        print(f"\nPerforming hyperparameter tuning for {analyzer.best_model}...")
        try:
            tuning_results = analyzer.hyperparameter_tuning(
                analyzer.best_model, cv_folds=3
            )
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    analyzer.plot_comparison('sklearn_guinier_comparison.png')
    
    return analyzer


if __name__ == "__main__":
    demo_sklearn_guinier() 
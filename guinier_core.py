import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import os


class GuinierAnalyzer:
    """
    Core class for Guinier analysis of SAXS data.
    Handles data loading, processing, and fitting without GUI dependencies.
    """
    
    def __init__(self):
        # Data storage
        self.q_data = None
        self.I_data = None
        self.dI_data = None
        self.filtered_indices = None
        
        # Processing parameters
        self.bg_value = 0.0
        self.norm_factor = 1.0
        self.snr_threshold = 3.0
        
        # Fitting parameters
        self.q_min_idx = 0
        self.q_max_idx = -1
        self.use_robust_fitting = True
        
        # Results
        self.Rg = None
        self.Rg_error = None
        self.I0 = None
        self.I0_error = None
        self.chi_squared = None
        self.r_squared = None
        self.fit_slope = None
        self.fit_intercept = None
        self.fit_covariance = None
        
    def load_data(self, filename):
        """
        Load SAXS data from file.
        
        Parameters:
        -----------
        filename : str
            Path to the data file (.grad, .txt, .csv, etc.)
            
        Returns:
        --------
        dict : Status information with keys 'success', 'message', 'n_points'
        """
        try:
            if filename.lower().endswith('.grad'):
                return self._load_grad_file(filename)
            else:
                return self._load_standard_file(filename)
        except Exception as e:
            return {'success': False, 'message': f"Failed to load data: {str(e)}", 'n_points': 0}
    
    def _load_grad_file(self, filename):
        """Load .grad file format"""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find where the data starts
        data_start = 0
        for i, line in enumerate(lines):
            if "q,I,dI" in line:
                data_start = i + 1
                break
        
        # Find where the data ends (before XML metadata)
        data_end = len(lines)
        for i, line in enumerate(lines[data_start:], start=data_start):
            if line.startswith('#HEADERS'):
                data_end = i
                break
        
        # Parse the data rows
        q_values = []
        i_values = []
        di_values = []
        
        for i in range(data_start, data_end):
            line = lines[i].strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    q = float(parts[0])
                    i_val = parts[1].strip()
                    
                    # Handle NaN values
                    if i_val.lower() == 'nan':
                        continue
                    
                    i = float(i_val)
                    di = float(parts[2]) if parts[2].strip().lower() != 'nan' else 0.0
                    
                    q_values.append(q)
                    i_values.append(i)
                    di_values.append(di)
                except (ValueError, IndexError):
                    continue
        
        if not q_values:
            raise ValueError("No valid data points found in the .grad file")
        
        self.q_data = np.array(q_values)
        self.I_data = np.array(i_values)
        self.dI_data = np.array(di_values)
        
        return self._finalize_data_loading(filename)
    
    def _load_standard_file(self, filename):
        """Load standard data file (txt, csv, etc.)"""
        data = pd.read_csv(filename, sep=None, engine='python', header=None)
        
        if data.shape[1] < 2:
            raise ValueError("Data file must have at least 2 columns (q and I)")
        
        self.q_data = data.iloc[:, 0].values
        self.I_data = data.iloc[:, 1].values
        
        # If there's a third column, it might be dI (error)
        if data.shape[1] >= 3:
            self.dI_data = data.iloc[:, 2].values
        else:
            self.dI_data = None
        
        return self._finalize_data_loading(filename)
    
    def _finalize_data_loading(self, filename):
        """Common finalization steps for data loading"""
        # Filter out any NaN values
        valid_idx = np.isfinite(self.q_data) & np.isfinite(self.I_data)
        if not np.any(valid_idx):
            raise ValueError("No valid data points found after filtering NaN values")
        
        self.q_data = self.q_data[valid_idx]
        self.I_data = self.I_data[valid_idx]
        if self.dI_data is not None:
            self.dI_data = self.dI_data[valid_idx]
        
        # Reset parameters
        self.filtered_indices = None
        self.q_min_idx = 0
        self.q_max_idx = len(self.q_data) - 1
        self.bg_value = 0.0
        self.norm_factor = 1.0
        self._reset_results()
        
        return {
            'success': True,
            'message': f"Loaded {len(self.q_data)} data points from {os.path.basename(filename)}",
            'n_points': len(self.q_data)
        }
    
    def _reset_results(self):
        """Reset all fitting results"""
        self.Rg = None
        self.Rg_error = None
        self.I0 = None
        self.I0_error = None
        self.chi_squared = None
        self.r_squared = None
        self.fit_slope = None
        self.fit_intercept = None
        self.fit_covariance = None
    
    def apply_corrections(self, bg_value=None, norm_factor=None, snr_threshold=None):
        """
        Apply background subtraction, normalization, and SNR filtering.
        
        Parameters:
        -----------
        bg_value : float, optional
            Background value to subtract
        norm_factor : float, optional
            Normalization factor
        snr_threshold : float, optional
            Minimum signal-to-noise ratio threshold
            
        Returns:
        --------
        dict : Status information with filtering results
        """
        if self.q_data is None or self.I_data is None:
            return {'success': False, 'message': "No data loaded", 'n_filtered': 0}
        
        # Update parameters if provided
        if bg_value is not None:
            self.bg_value = bg_value
        if norm_factor is not None:
            self.norm_factor = norm_factor
        if snr_threshold is not None:
            self.snr_threshold = snr_threshold
        
        try:
            # Apply SNR filtering if we have error data
            if self.dI_data is not None:
                I_corrected = (self.I_data - self.bg_value) / self.norm_factor
                SNR = I_corrected / (self.dI_data / self.norm_factor)
                
                # Find indices where SNR >= threshold
                self.filtered_indices = np.where(SNR >= self.snr_threshold)[0]
                
                if len(self.filtered_indices) == 0:
                    self.filtered_indices = np.arange(len(self.q_data))
                    message = f"No data points meet SNR threshold of {self.snr_threshold}. Using all data."
                else:
                    message = f"Applied SNR filter: {len(self.filtered_indices)} of {len(self.q_data)} points retained."
            else:
                # If no error data, use all points
                self.filtered_indices = np.arange(len(self.q_data))
                message = "Applied background subtraction and normalization (no error data for SNR filtering)."
            
            # Update fit range after filtering
            self.q_min_idx = 0
            self.q_max_idx = len(self.filtered_indices) - 1
            
            return {
                'success': True,
                'message': message,
                'n_filtered': len(self.filtered_indices),
                'n_total': len(self.q_data)
            }
            
        except Exception as e:
            return {'success': False, 'message': f"Error applying corrections: {str(e)}", 'n_filtered': 0}
    
    def set_fit_range(self, q_min_idx, q_max_idx):
        """
        Set the fitting range by index.
        
        Parameters:
        -----------
        q_min_idx : int
            Minimum index for fitting range
        q_max_idx : int
            Maximum index for fitting range (can be negative for reverse indexing)
            
        Returns:
        --------
        dict : Status information
        """
        if self.q_data is None:
            return {'success': False, 'message': "No data loaded"}
        
        if self.filtered_indices is None:
            self.filtered_indices = np.arange(len(self.q_data))
        
        try:
            # Handle negative indexing
            if q_max_idx < 0:
                q_max_idx = len(self.filtered_indices) + q_max_idx
            
            # Validate range
            if (q_min_idx < 0 or q_min_idx >= len(self.filtered_indices) or 
                q_max_idx < 0 or q_max_idx >= len(self.filtered_indices) or 
                q_min_idx >= q_max_idx):
                raise ValueError("Invalid index range")
            
            self.q_min_idx = q_min_idx
            self.q_max_idx = q_max_idx
            
            return {
                'success': True,
                'message': f"Set fitting range: {self.q_min_idx} to {self.q_max_idx}",
                'q_min_idx': self.q_min_idx,
                'q_max_idx': self.q_max_idx
            }
            
        except Exception as e:
            return {'success': False, 'message': f"Invalid fit range: {str(e)}"}
    
    def auto_range(self, q_rg_limit=1.3):
        """
        Automatically determine the fitting range based on q·Rg ≤ limit.
        
        Parameters:
        -----------
        q_rg_limit : float
            Maximum allowed q·Rg value (default: 1.3)
            
        Returns:
        --------
        dict : Status information with estimated Rg and range
        """
        if self.q_data is None or self.I_data is None:
            return {'success': False, 'message': "No data loaded"}
        
        if self.filtered_indices is None:
            self.filtered_indices = np.arange(len(self.q_data))
        
        try:
            # Get filtered data
            q_filtered = self.q_data[self.filtered_indices]
            I_filtered = (self.I_data[self.filtered_indices] - self.bg_value) / self.norm_factor
            
            # Estimate Rg using full range
            q_sq = q_filtered**2
            ln_I = np.log(I_filtered)
            valid_idx = np.isfinite(ln_I)
            
            if not np.any(valid_idx):
                raise ValueError("No valid points for fitting")
            
            # Weighted fit if error data available
            if self.dI_data is not None:
                dI_filtered = self.dI_data[self.filtered_indices] / self.norm_factor
                weights = I_filtered[valid_idx]**2 / (dI_filtered[valid_idx]**2)
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
                params, _ = np.polyfit(q_sq[valid_idx], ln_I[valid_idx], 1, w=weights, cov=True)
            else:
                params, _ = np.polyfit(q_sq[valid_idx], ln_I[valid_idx], 1, cov=True)
            
            slope = params[0]
            initial_Rg = np.sqrt(-3 * slope)
            
            if initial_Rg <= 0:
                raise ValueError("Initial Rg estimate is negative or zero")
            
            # Find range where q*Rg <= limit
            q_Rg = q_filtered * initial_Rg
            valid_pts = np.where(q_Rg <= q_rg_limit)[0]
            
            if len(valid_pts) > 0:
                self.q_min_idx = 0
                self.q_max_idx = valid_pts[-1]
                
                # Check if we have enough points
                if self.q_max_idx - self.q_min_idx < 10:
                    # Try relaxing the criterion
                    extended_pts = np.where(q_Rg <= 1.5)[0]
                    if len(extended_pts) >= 10:
                        self.q_max_idx = extended_pts[-1]
                        message = f"Extended range to q·Rg ≤ 1.5 to include more points"
                    else:
                        message = f"Auto set range where q·Rg ≤ {q_rg_limit}"
                else:
                    message = f"Auto set range where q·Rg ≤ {q_rg_limit}"
                
                return {
                    'success': True,
                    'message': message,
                    'initial_Rg': initial_Rg,
                    'q_min_idx': self.q_min_idx,
                    'q_max_idx': self.q_max_idx
                }
            else:
                return {'success': False, 'message': f"Could not find valid range with q·Rg ≤ {q_rg_limit}"}
                
        except Exception as e:
            return {'success': False, 'message': f"Failed to auto-set range: {str(e)}"}
    
    def perform_fit(self, use_robust=None):
        """
        Perform Guinier fitting on the selected data range.
        
        Parameters:
        -----------
        use_robust : bool, optional
            Whether to use robust fitting methods
            
        Returns:
        --------
        dict : Fitting results and status
        """
        if self.q_data is None or self.I_data is None:
            return {'success': False, 'message': "No data loaded"}
        
        if self.filtered_indices is None:
            self.filtered_indices = np.arange(len(self.q_data))
        
        if use_robust is not None:
            self.use_robust_fitting = use_robust
        
        try:
            # Get filtered data
            q_filtered = self.q_data[self.filtered_indices]
            I_filtered = self.I_data[self.filtered_indices]
            
            # Get data in range for fitting
            q_range = q_filtered[self.q_min_idx:self.q_max_idx+1]
            I_range = I_filtered[self.q_min_idx:self.q_max_idx+1]
            
            # Apply corrections
            I_corrected = (I_range - self.bg_value) / self.norm_factor
            
            # Create Guinier plot data
            q_sq = q_range**2
            ln_I = np.log(I_corrected)
            valid_idx = np.isfinite(ln_I)
            
            if not np.any(valid_idx):
                raise ValueError("No valid points for fitting")
            
            # Perform fitting
            fit_result = self._perform_linear_fit(q_sq[valid_idx], ln_I[valid_idx], 
                                                 I_corrected[valid_idx])
            
            if not fit_result['success']:
                return fit_result
            
            # Extract results
            self.fit_intercept = fit_result['intercept']
            self.fit_slope = fit_result['slope']
            self.fit_covariance = fit_result['covariance']
            
            # Calculate Rg and I0
            if self.fit_slope >= 0:
                raise ValueError("Slope is positive, cannot compute Rg (expected negative slope)")
            
            self.Rg = np.sqrt(-3 * self.fit_slope)
            self.I0 = np.exp(self.fit_intercept)
            
            # Calculate errors
            if self.fit_covariance is not None:
                slope_err, intercept_err = np.sqrt(np.diag(self.fit_covariance))
                self.Rg_error = self.Rg * 0.5 * slope_err / abs(self.fit_slope)
                self.I0_error = self.I0 * intercept_err
            else:
                self.Rg_error = 0.0
                self.I0_error = 0.0
            
            # Calculate fit quality metrics
            self._calculate_fit_quality(q_sq[valid_idx], ln_I[valid_idx], 
                                       I_corrected[valid_idx])
            
            # Physical reasonability check
            max_q_rg = q_range[valid_idx][-1] * self.Rg
            warning = ""
            if max_q_rg > 1.3:
                warning = f"Warning: q·Rg exceeds 1.3 (max q·Rg = {max_q_rg:.2f}). Results may be unreliable."
            
            return {
                'success': True,
                'message': f"Guinier fit completed: Rg = {self.Rg:.2f} ± {self.Rg_error:.2f} Å",
                'warning': warning,
                'Rg': self.Rg,
                'Rg_error': self.Rg_error,
                'I0': self.I0,
                'I0_error': self.I0_error,
                'r_squared': self.r_squared,
                'chi_squared': self.chi_squared,
                'max_q_rg': max_q_rg
            }
            
        except Exception as e:
            return {'success': False, 'message': f"Fitting failed: {str(e)}"}
    
    def _perform_linear_fit(self, q_sq, ln_I, I_corrected):
        """Perform linear fitting with optional robust methods"""
        # Get weights if error data available
        use_weights = False
        weights = None
        
        if self.dI_data is not None and self.filtered_indices is not None:
            dI_filtered = self.dI_data[self.filtered_indices]
            dI_range = dI_filtered[self.q_min_idx:self.q_max_idx+1] / self.norm_factor
            valid_range_idx = np.isfinite(ln_I)
            
            if np.any(valid_range_idx):
                weights = I_corrected[valid_range_idx]**2 / (dI_range[valid_range_idx]**2)
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
                use_weights = True
        
        # Try robust fitting if requested
        if self.use_robust_fitting:
            try:
                return self._robust_fit(q_sq, ln_I, weights, use_weights)
            except ImportError:
                # Fall back to standard fitting
                pass
            except Exception as e:
                # If robust fitting fails, fall back to standard
                pass
        
        # Standard polyfit
        if use_weights:
            params, pcov = np.polyfit(q_sq, ln_I, 1, w=weights, cov=True)
        else:
            params, pcov = np.polyfit(q_sq, ln_I, 1, cov=True)
        
        return {
            'success': True,
            'slope': params[0],
            'intercept': params[1],
            'covariance': pcov
        }
    
    def _robust_fit(self, q_sq, ln_I, weights, use_weights):
        """Perform robust fitting using available methods"""
        if use_weights:
            # Use Huber regressor for weighted robust regression
            try:
                from sklearn.linear_model import HuberRegressor
                huber = HuberRegressor(epsilon=1.35, max_iter=100)
                X = q_sq.reshape(-1, 1)
                y = ln_I
                norm_weights = weights / np.sum(weights)
                huber.fit(X, y, sample_weight=norm_weights)
                
                slope = huber.coef_[0]
                intercept = huber.intercept_
                
                # Estimate covariance
                y_pred = intercept + slope * X.flatten()
                residuals = y - y_pred
                weighted_residuals = residuals * np.sqrt(norm_weights)
                mse = np.mean(weighted_residuals**2)
                X_mean = np.mean(X)
                X_var = np.sum(norm_weights * (X.flatten() - X_mean)**2)
                slope_var = mse / X_var
                intercept_var = mse * (1/len(X) + X_mean**2 / X_var)
                pcov = np.array([[intercept_var, 0], [0, slope_var]])
                
                return {
                    'success': True,
                    'slope': slope,
                    'intercept': intercept,
                    'covariance': pcov
                }
            except ImportError:
                raise ImportError("sklearn not available for robust fitting")
        else:
            # Use Theil-Sen estimator
            slope, intercept, _, _ = stats.theilslopes(ln_I, q_sq)
            
            # Estimate covariance
            y_pred = intercept + slope * q_sq
            residuals = ln_I - y_pred
            mse = np.mean(residuals**2)
            X_mean = np.mean(q_sq)
            X_var = np.sum((q_sq - X_mean)**2)
            slope_var = mse / X_var
            intercept_var = mse * (1/len(q_sq) + X_mean**2 / X_var)
            pcov = np.array([[intercept_var, 0], [0, slope_var]])
            
            return {
                'success': True,
                'slope': slope,
                'intercept': intercept,
                'covariance': pcov
            }
    
    def _calculate_fit_quality(self, q_sq, ln_I, I_corrected):
        """Calculate fit quality metrics"""
        # Calculate fit
        y_fit = self.fit_intercept + self.fit_slope * q_sq
        residuals = ln_I - y_fit
        
        # R-squared
        ss_total = np.sum((ln_I - np.mean(ln_I))**2)
        ss_residual = np.sum(residuals**2)
        self.r_squared = 1 - (ss_residual / ss_total)
        
        # Reduced chi-squared
        n = len(q_sq)
        p = 2  # Two parameters
        
        if self.dI_data is not None and self.filtered_indices is not None:
            dI_filtered = self.dI_data[self.filtered_indices]
            dI_range = dI_filtered[self.q_min_idx:self.q_max_idx+1] / self.norm_factor
            valid_range_idx = np.isfinite(ln_I)
            
            if np.any(valid_range_idx):
                weights = I_corrected[valid_range_idx]**2 / (dI_range[valid_range_idx]**2)
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
                chi_sq = np.sum(weights * residuals**2) / (n - p)
            else:
                chi_sq = np.sum(residuals**2) / (n - p)
        else:
            chi_sq = np.sum(residuals**2) / (n - p)
        
        self.chi_squared = chi_sq
    
    def get_processed_data(self):
        """
        Get processed data for plotting and analysis.
        
        Returns:
        --------
        dict : Processed data arrays
        """
        if self.q_data is None or self.I_data is None:
            return None
        
        # Original data
        I_corrected_all = (self.I_data - self.bg_value) / self.norm_factor
        
        result = {
            'q_original': self.q_data,
            'I_original': self.I_data,
            'I_corrected_all': I_corrected_all,
            'dI_original': self.dI_data,
            'bg_value': self.bg_value,
            'norm_factor': self.norm_factor,
            'snr_threshold': self.snr_threshold
        }
        
        # Filtered data
        if self.filtered_indices is not None:
            q_filtered = self.q_data[self.filtered_indices]
            I_filtered = self.I_data[self.filtered_indices]
            I_corrected_filtered = (I_filtered - self.bg_value) / self.norm_factor
            
            result.update({
                'q_filtered': q_filtered,
                'I_filtered': I_filtered,
                'I_corrected_filtered': I_corrected_filtered,
                'filtered_indices': self.filtered_indices
            })
            
            if self.dI_data is not None:
                dI_filtered = self.dI_data[self.filtered_indices]
                result['dI_filtered'] = dI_filtered
            
            # Fit range data
            if self.q_min_idx is not None and self.q_max_idx is not None:
                q_range = q_filtered[self.q_min_idx:self.q_max_idx+1]
                I_range = I_filtered[self.q_min_idx:self.q_max_idx+1]
                I_corrected_range = (I_range - self.bg_value) / self.norm_factor
                
                result.update({
                    'q_range': q_range,
                    'I_range': I_range,
                    'I_corrected_range': I_corrected_range,
                    'q_min_idx': self.q_min_idx,
                    'q_max_idx': self.q_max_idx
                })
                
                if self.dI_data is not None:
                    dI_range = dI_filtered[self.q_min_idx:self.q_max_idx+1]
                    result['dI_range'] = dI_range
        
        return result
    
    def get_fit_results(self):
        """
        Get fitting results in a structured format.
        
        Returns:
        --------
        dict : Fitting results or None if no fit performed
        """
        if self.Rg is None or self.I0 is None:
            return None
        
        # Get data for q*Rg calculation
        data = self.get_processed_data()
        max_q_rg = None
        if data and 'q_range' in data:
            max_q_rg = data['q_range'][-1] * self.Rg
        
        return {
            'Rg': self.Rg,
            'Rg_error': self.Rg_error,
            'I0': self.I0,
            'I0_error': self.I0_error,
            'r_squared': self.r_squared,
            'chi_squared': self.chi_squared,
            'fit_slope': self.fit_slope,
            'fit_intercept': self.fit_intercept,
            'max_q_rg': max_q_rg,
            'use_robust_fitting': self.use_robust_fitting
        }
    
    def generate_fit_curve(self, q_values=None):
        """
        Generate Guinier fit curve for given q values.
        
        Parameters:
        -----------
        q_values : array-like, optional
            q values to generate curve for. If None, uses fit range.
            
        Returns:
        --------
        dict : Fit curve data
        """
        if self.Rg is None or self.I0 is None:
            return None
        
        if q_values is None:
            data = self.get_processed_data()
            if data and 'q_range' in data:
                q_values = data['q_range']
            else:
                return None
        
        # Generate Guinier curve: I(q) = I0 * exp(-q^2 * Rg^2 / 3)
        I_fit = self.I0 * np.exp(-(q_values**2) * self.Rg**2 / 3)
        
        # Generate linear fit for Guinier plot: ln(I) = ln(I0) - q^2 * Rg^2 / 3
        q_sq = q_values**2
        ln_I_fit = self.fit_intercept + self.fit_slope * q_sq
        
        return {
            'q_values': q_values,
            'I_fit': I_fit,
            'q_sq': q_sq,
            'ln_I_fit': ln_I_fit
        }
    
    def save_results(self, filename):
        """
        Save fitting results to file.
        
        Parameters:
        -----------
        filename : str
            Output filename (CSV format)
            
        Returns:
        --------
        dict : Status information
        """
        if self.Rg is None or self.I0 is None:
            return {'success': False, 'message': "No fitting results to save"}
        
        try:
            # Prepare results data
            results_data = self._prepare_results_data()
            
            # Save results
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(filename, index=False)
            
            # Save fitted data
            data_filename = os.path.splitext(filename)[0] + "_data.csv"
            data_result = self._save_fitted_data(data_filename)
            
            if data_result['success']:
                return {
                    'success': True,
                    'message': f"Results saved to {filename}",
                    'data_file': data_filename
                }
            else:
                return {
                    'success': True,
                    'message': f"Results saved to {filename} (data save failed: {data_result['message']})"
                }
                
        except Exception as e:
            return {'success': False, 'message': f"Failed to save results: {str(e)}"}
    
    def _prepare_results_data(self):
        """Prepare results data for saving"""
        parameters = ["Radius of Gyration (Rg)", "Zero-angle Intensity (I₀)"]
        values = [f"{self.Rg:.4f}", f"{self.I0:.4e}"]
        errors = [f"{self.Rg_error:.4f}", f"{self.I0_error:.4e}"]
        
        # Add validation metrics
        if self.r_squared is not None:
            parameters.append("R² (Goodness of fit)")
            values.append(f"{self.r_squared:.4f}")
            errors.append("")
        
        if self.chi_squared is not None:
            parameters.append("χ²ᵣₑₙ (Reduced chi-squared)")
            values.append(f"{self.chi_squared:.4f}")
            errors.append("")
        
        # Add fitting range info
        data = self.get_processed_data()
        if data and 'q_range' in data:
            q_range = data['q_range']
            max_q_rg = q_range[-1] * self.Rg
            
            parameters.extend([
                "Fitting Range (q·Rg max)",
                "q min (Å⁻¹)",
                "q max (Å⁻¹)"
            ])
            values.extend([
                f"{max_q_rg:.4f}",
                f"{q_range[0]:.6f}",
                f"{q_range[-1]:.6f}"
            ])
            errors.extend(["", "", ""])
        
        # Add processing parameters
        parameters.extend(["Background Value", "Normalization Factor", "SNR Threshold"])
        values.extend([f"{self.bg_value:.4f}", f"{self.norm_factor:.4f}", f"{self.snr_threshold:.1f}"])
        errors.extend(["", "", ""])
        
        return {
            "Parameter": parameters,
            "Value": values,
            "Error": errors
        }
    
    def _save_fitted_data(self, filename):
        """Save fitted data points"""
        try:
            data = self.get_processed_data()
            if not data or 'q_range' not in data:
                return {'success': False, 'message': "No fitted data available"}
            
            # Prepare data dictionary
            data_dict = {
                "q (Å⁻¹)": data['q_range'],
                "I_raw": data['I_range'],
                "I_corrected": data['I_corrected_range'],
                "ln(I)": np.log(data['I_corrected_range']),
                "q² (Å⁻²)": data['q_range']**2,
                "q·Rg": data['q_range'] * self.Rg
            }
            
            # Add Guinier fit
            fit_curve = self.generate_fit_curve(data['q_range'])
            if fit_curve:
                data_dict["Guinier_fit"] = fit_curve['I_fit']
            
            # Add error data if available
            if 'dI_range' in data:
                dI_corrected = data['dI_range'] / self.norm_factor
                data_dict["dI"] = dI_corrected
                data_dict["SNR"] = data['I_corrected_range'] / dI_corrected
            
            # Save to CSV
            data_df = pd.DataFrame(data_dict)
            data_df.to_csv(filename, index=False)
            
            return {'success': True, 'message': f"Data saved to {filename}"}
            
        except Exception as e:
            return {'success': False, 'message': f"Failed to save data: {str(e)}"} 
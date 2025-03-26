import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from scipy.optimize import curve_fit
import os


class GuinierAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Guinier Analysis for SAXS Data")
        self.root.geometry("1200x800")
        
        # Data variables
        self.q_data = None
        self.I_data = None
        self.dI_data = None  # Add error data storage
        self.filtered_indices = None  # To store indices after SNR filtering
        self.bg_value = 0.0
        self.norm_factor = 1.0
        self.snr_threshold = 3.0  # Default SNR threshold
        self.q_min_idx = 0
        self.q_max_idx = -1
        self.Rg = None
        self.Rg_error = None
        self.I0 = None
        self.I0_error = None
        self.chi_squared = None  # Reduced chi-squared
        self.r_squared = None    # R-squared for goodness of fit
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Data loading
        load_frame = ttk.LabelFrame(control_frame, text="Data Loading")
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load SAXS Data", command=self.load_data).pack(padx=5, pady=5)
        ttk.Label(load_frame, text="Data Format: .grad, or q, I columns").pack(padx=5, pady=2)
        
        # Data Processing
        process_frame = ttk.LabelFrame(control_frame, text="Data Processing")
        process_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Background subtraction
        bg_frame = ttk.Frame(process_frame)
        bg_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(bg_frame, text="Background:").pack(side=tk.LEFT, padx=5)
        self.bg_var = tk.StringVar(value="0.0")
        ttk.Entry(bg_frame, textvariable=self.bg_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Normalization
        norm_frame = ttk.Frame(process_frame)
        norm_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(norm_frame, text="Norm Factor:").pack(side=tk.LEFT, padx=5)
        self.norm_var = tk.StringVar(value="1.0")
        ttk.Entry(norm_frame, textvariable=self.norm_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # SNR Filtering
        snr_frame = ttk.Frame(process_frame)
        snr_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(snr_frame, text="Min SNR (I/σ):").pack(side=tk.LEFT, padx=5)
        self.snr_var = tk.StringVar(value="3.0")
        ttk.Entry(snr_frame, textvariable=self.snr_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(process_frame, text="Apply Corrections", command=self.apply_corrections).pack(padx=5, pady=5)
        
        # Fitting range
        fit_range_frame = ttk.LabelFrame(control_frame, text="Fitting Range")
        fit_range_frame.pack(fill=tk.X, padx=5, pady=5)
        
        q_min_frame = ttk.Frame(fit_range_frame)
        q_min_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(q_min_frame, text="q_min index:").pack(side=tk.LEFT, padx=5)
        self.q_min_var = tk.StringVar(value="0")
        ttk.Entry(q_min_frame, textvariable=self.q_min_var, width=10).pack(side=tk.LEFT, padx=5)
        
        q_max_frame = ttk.Frame(fit_range_frame)
        q_max_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(q_max_frame, text="q_max index:").pack(side=tk.LEFT, padx=5)
        self.q_max_var = tk.StringVar(value="-1")
        ttk.Entry(q_max_frame, textvariable=self.q_max_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(fit_range_frame, text="Set Range", command=self.set_fit_range).pack(padx=5, pady=5)
        ttk.Button(fit_range_frame, text="Auto Range (q·Rg ≤ 1.3)", command=self.auto_range).pack(padx=5, pady=5)
        
        # Fitting
        fit_frame = ttk.LabelFrame(control_frame, text="Guinier Fitting")
        fit_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Robust fitting option
        robust_frame = ttk.Frame(fit_frame)
        robust_frame.pack(fill=tk.X, padx=5, pady=5)
        self.robust_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(robust_frame, text="Use robust fitting (reduce outlier impact)", 
                       variable=self.robust_var).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(fit_frame, text="Perform Guinier Fit", command=self.perform_fit).pack(padx=5, pady=5)
        ttk.Button(fit_frame, text="Save Results", command=self.save_results).pack(padx=5, pady=5)
        
        # Results
        self.result_text = tk.Text(control_frame, height=10, width=40, state="disabled")
        self.result_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Right panel (plots)
        plot_frame = ttk.LabelFrame(main_frame, text="Plots")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create plots - now with 3 subplots (data, guinier, residuals)
        self.fig = Figure(figsize=(10, 8))
        
        # Raw data plot
        self.ax1 = self.fig.add_subplot(311)
        self.ax1.set_title("SAXS Data")
        self.ax1.set_xlabel("q (Å⁻¹)")
        self.ax1.set_ylabel("I(q)")
        self.ax1.grid(True)
        
        # Guinier plot
        self.ax2 = self.fig.add_subplot(312)
        self.ax2.set_title("Guinier Plot: ln(I) vs q²")
        self.ax2.set_xlabel("q² (Å⁻²)")
        self.ax2.set_ylabel("ln(I)")
        self.ax2.grid(True)
        
        # Residuals plot
        self.ax3 = self.fig.add_subplot(313)
        self.ax3.set_title("Residuals")
        self.ax3.set_xlabel("q² (Å⁻²)")
        self.ax3.set_ylabel("Residuals")
        self.ax3.grid(True)
        self.ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        self.fig.tight_layout()
        
        # Embed plots in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    def load_data(self):
        filename = filedialog.askopenfilename(
            title="Select SAXS Data File",
            filetypes=[("GRAD Files", "*.grad"), ("Text Files", "*.txt"), 
                       ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            # Check if it's a .grad file
            if filename.lower().endswith('.grad'):
                # Parse .grad file format
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
                            # Skip malformed lines
                            continue
                
                if not q_values:
                    raise ValueError("No valid data points found in the .grad file")
                
                self.q_data = np.array(q_values)
                self.I_data = np.array(i_values)
                self.dI_data = np.array(di_values)  # Store error values for future use
                
            else:
                # Try to load data with pandas
                data = pd.read_csv(filename, sep=None, engine='python', header=None)
                
                # If more than 2 columns, use the first two
                if data.shape[1] >= 2:
                    self.q_data = data.iloc[:, 0].values
                    self.I_data = data.iloc[:, 1].values
                    
                    # If there's a third column, it might be dI (error)
                    if data.shape[1] >= 3:
                        self.dI_data = data.iloc[:, 2].values
                    else:
                        self.dI_data = None
                else:
                    raise ValueError("Data file must have at least 2 columns (q and I)")
            
            # Filter out any NaN values
            valid_idx = np.isfinite(self.q_data) & np.isfinite(self.I_data)
            if not np.any(valid_idx):
                raise ValueError("No valid data points found after filtering NaN values")
                
            self.q_data = self.q_data[valid_idx]
            self.I_data = self.I_data[valid_idx]
            if hasattr(self, 'dI_data') and self.dI_data is not None:
                self.dI_data = self.dI_data[valid_idx]
                
            # Reset fit range
            self.q_min_idx = 0
            self.q_max_idx = len(self.q_data) - 1
            self.q_min_var.set(str(self.q_min_idx))
            self.q_max_var.set(str(self.q_max_idx))
            
            # Reset corrections
            self.bg_value = 0.0
            self.norm_factor = 1.0
            self.bg_var.set("0.0")
            self.norm_var.set("1.0")
            
            # Update plots
            self.update_plots()
            messagebox.showinfo("Success", f"Loaded {len(self.q_data)} data points from {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def apply_corrections(self):
        if self.q_data is None or self.I_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            self.bg_value = float(self.bg_var.get())
            self.norm_factor = float(self.norm_var.get())
            self.snr_threshold = float(self.snr_var.get())
            
            # Apply SNR filtering if we have error data
            if hasattr(self, 'dI_data') and self.dI_data is not None:
                I_corrected = (self.I_data - self.bg_value) / self.norm_factor
                SNR = I_corrected / (self.dI_data / self.norm_factor)
                
                # Find indices where SNR >= threshold
                self.filtered_indices = np.where(SNR >= self.snr_threshold)[0]
                
                if len(self.filtered_indices) == 0:
                    messagebox.showwarning("Warning", 
                                          f"No data points meet SNR threshold of {self.snr_threshold}. Using all data.")
                    self.filtered_indices = np.arange(len(self.q_data))
                else:
                    messagebox.showinfo("Info", 
                                       f"Applied SNR filter: {len(self.filtered_indices)} of {len(self.q_data)} points retained.")
            else:
                # If no error data, use all points
                self.filtered_indices = np.arange(len(self.q_data))
            
            # Update fit range after filtering
            self.q_min_idx = 0
            self.q_max_idx = len(self.filtered_indices) - 1
            self.q_min_var.set(str(self.q_min_idx))
            self.q_max_var.set(str(self.q_max_idx))
            
            self.update_plots()
            messagebox.showinfo("Success", "Applied background subtraction, normalization, and SNR filtering")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid correction parameters")
    
    def set_fit_range(self):
        if self.q_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            q_min_idx = int(self.q_min_var.get())
            q_max_idx = int(self.q_max_var.get())
            
            # Handle negative indexing
            if q_max_idx < 0:
                q_max_idx = len(self.q_data) + q_max_idx
                
            if q_min_idx < 0 or q_min_idx >= len(self.q_data) or q_max_idx < 0 or q_max_idx >= len(self.q_data) or q_min_idx >= q_max_idx:
                raise ValueError("Invalid index range")
                
            self.q_min_idx = q_min_idx
            self.q_max_idx = q_max_idx
            
            self.update_plots()
            messagebox.showinfo("Success", f"Set fitting range: {self.q_min_idx} to {self.q_max_idx}")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid fit range: {str(e)}")
    
    def auto_range(self):
        if self.q_data is None or self.I_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        # Need to estimate Rg first
        try:
            # Use filtered data
            if self.filtered_indices is None:
                # If not filtered yet, use all data
                self.filtered_indices = np.arange(len(self.q_data))
                
            q_filtered = self.q_data[self.filtered_indices]
            I_filtered = (self.I_data[self.filtered_indices] - self.bg_value) / self.norm_factor
            
            # Use full range for initial estimate
            q_sq = q_filtered**2
            ln_I = np.log(I_filtered)
            
            # Fit to get rough Rg estimate
            valid_idx = np.isfinite(ln_I)
            if not np.any(valid_idx):
                raise ValueError("No valid points for fitting")
                
            # If we have error data, use weighted fit for initial estimate
            if hasattr(self, 'dI_data') and self.dI_data is not None and self.filtered_indices is not None:
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
                
            # Find range where q*Rg <= 1.3
            q_Rg = q_filtered * initial_Rg
            valid_pts = np.where(q_Rg <= 1.3)[0]
            
            if len(valid_pts) > 0:
                # Start from 0 and go to the furthest point where q*Rg <= 1.3
                self.q_min_idx = 0
                self.q_max_idx = valid_pts[-1]
                
                # Check if we have enough points (need at least 10 for reliable fit)
                if self.q_max_idx - self.q_min_idx < 10:
                    # If not enough points, try relaxing the criterion a bit
                    q_Rg = q_filtered * initial_Rg
                    extended_pts = np.where(q_Rg <= 1.5)[0]  # Try with 1.5 instead of 1.3
                    
                    if len(extended_pts) >= 10:
                        self.q_max_idx = extended_pts[-1]
                        messagebox.showinfo("Note", 
                                          f"Extended range to q·Rg ≤ 1.5 to include more points")
                
                self.q_min_var.set(str(self.q_min_idx))
                self.q_max_var.set(str(self.q_max_idx))
                
                self.update_plots()
                messagebox.showinfo("Success", 
                                  f"Auto set range where q·Rg ≤ 1.3 (Estimated Rg: {initial_Rg:.2f} Å)")
            else:
                messagebox.showwarning("Warning", "Could not find valid range with q·Rg ≤ 1.3")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to auto-set range: {str(e)}")
    
    def guinier_func(self, q_sq, I0, Rg):
        return np.log(I0) - (q_sq * Rg**2 / 3)
    
    def perform_fit(self):
        if self.q_data is None or self.I_data is None:
            messagebox.showwarning("Warning", "No data loaded")
            return
            
        try:
            # Make sure filtered indices are available
            if self.filtered_indices is None:
                self.filtered_indices = np.arange(len(self.q_data))
                
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
            
            # Check for valid data
            valid_idx = np.isfinite(ln_I)
            if not np.any(valid_idx):
                raise ValueError("No valid points for fitting")
            
            # Initialize variables to store fit results
            intercept, slope = None, None
            pcov = None
            
            # Get weights from error bars if available
            use_weights = False
            weights = None
            if hasattr(self, 'dI_data') and self.dI_data is not None:
                dI_filtered = self.dI_data[self.filtered_indices]
                dI_range = dI_filtered[self.q_min_idx:self.q_max_idx+1] / self.norm_factor
                # For weighted linear fit, weight = 1/σ²
                # For ln(I), error propagation gives σ_ln(I) = σ_I/I
                weights = I_corrected[valid_idx]**2 / (dI_range[valid_idx]**2)
                # Handle infinite or zero weights
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
                use_weights = True
            
            # Use robust fitting if selected
            use_robust = self.robust_var.get()
            
            if use_robust:
                try:
                    from scipy import stats
                    # Use robust linear regression (e.g., Theil-Sen or RANSAC)
                    if use_weights:
                        # For weighted robust regression, we use Huber regressor
                        from sklearn.linear_model import HuberRegressor
                        huber = HuberRegressor(epsilon=1.35, max_iter=100)
                        X = q_sq[valid_idx].reshape(-1, 1)
                        y = ln_I[valid_idx]
                        # Normalize weights to sum to 1
                        norm_weights = weights / np.sum(weights)
                        huber.fit(X, y, sample_weight=norm_weights)
                        slope = huber.coef_[0]
                        intercept = huber.intercept_
                        
                        # Estimate covariance using weighted residuals
                        y_pred = intercept + slope * X.flatten()
                        residuals = y - y_pred
                        weighted_residuals = residuals * np.sqrt(norm_weights)
                        mse = np.mean(weighted_residuals**2)
                        X_mean = np.mean(X)
                        X_var = np.sum(norm_weights * (X.flatten() - X_mean)**2)
                        slope_var = mse / X_var
                        intercept_var = mse * (1/len(X) + X_mean**2 / X_var)
                        pcov = np.array([[intercept_var, 0], [0, slope_var]])
                    else:
                        # Theil-Sen estimator (more robust than OLS)
                        slope, intercept, _, _ = stats.theilslopes(ln_I[valid_idx], q_sq[valid_idx])
                        
                        # Estimate covariance
                        y_pred = intercept + slope * q_sq[valid_idx]
                        residuals = ln_I[valid_idx] - y_pred
                        mse = np.mean(residuals**2)
                        X = q_sq[valid_idx]
                        X_mean = np.mean(X)
                        X_var = np.sum((X - X_mean)**2)
                        slope_var = mse / X_var
                        intercept_var = mse * (1/len(X) + X_mean**2 / X_var)
                        pcov = np.array([[intercept_var, 0], [0, slope_var]])
                except ImportError:
                    messagebox.showwarning("Warning", 
                                          "Robust fitting requires scipy.stats and sklearn. Using standard fit.")
                    use_robust = False
            
            # Use standard polyfit if robust fitting is not used or failed
            if not use_robust:
                if use_weights:
                    # Weighted linear fit
                    params, pcov = np.polyfit(q_sq[valid_idx], ln_I[valid_idx], 1, w=weights, cov=True)
                else:
                    # Unweighted linear fit
                    params, pcov = np.polyfit(q_sq[valid_idx], ln_I[valid_idx], 1, cov=True)
                slope, intercept = params[0], params[1]
            
            # Calculate Rg and I0
            if slope >= 0:
                raise ValueError("Slope is positive, cannot compute Rg (expected negative slope)")
                
            self.Rg = np.sqrt(-3 * slope)
            self.I0 = np.exp(intercept)
            
            # Calculate errors
            if pcov is not None:
                if isinstance(pcov, np.ndarray) and pcov.shape == (2, 2):
                    slope_err, intercept_err = np.sqrt(np.diag(pcov))
                    self.Rg_error = self.Rg * 0.5 * slope_err / abs(slope)
                    self.I0_error = self.I0 * intercept_err
                else:
                    self.Rg_error = 0.0
                    self.I0_error = 0.0
            else:
                self.Rg_error = 0.0
                self.I0_error = 0.0
            
            # Calculate fit quality metrics
            y_fit = intercept + slope * q_sq[valid_idx]
            residuals = ln_I[valid_idx] - y_fit
            
            # Calculate R-squared
            ss_total = np.sum((ln_I[valid_idx] - np.mean(ln_I[valid_idx]))**2)
            ss_residual = np.sum(residuals**2)
            self.r_squared = 1 - (ss_residual / ss_total)
            
            # Calculate reduced chi-squared
            # For weighted fit, chi^2 = sum(w_i * (y_i - f_i)^2) / (n - p)
            # where n is number of points, p is number of parameters (2 for line)
            n = len(q_sq[valid_idx])
            p = 2  # Two parameters: slope and intercept
            if use_weights:
                chi_sq = np.sum(weights * residuals**2) / (n - p)
            else:
                # For unweighted fit, assume uniform errors
                chi_sq = np.sum(residuals**2) / (n - p)
            self.chi_squared = chi_sq
            
            # Check physical reasonability
            max_q_rg = q_range[valid_idx][-1] * self.Rg
            if max_q_rg > 1.3:
                messagebox.showwarning("Warning", 
                                      f"q·Rg exceeds 1.3 (max q·Rg = {max_q_rg:.2f}). Results may be unreliable.")
            
            # Update plots with fit
            self.update_plots(show_fit=True, fit_params=(intercept, slope))
            
            # Update results
            self.update_results()
            
            messagebox.showinfo("Success", f"Guinier fit completed: Rg = {self.Rg:.2f} ± {self.Rg_error:.2f} Å")
            
        except Exception as e:
            messagebox.showerror("Error", f"Fitting failed: {str(e)}")
            
    def update_plots(self, show_fit=False, fit_params=None):
        if self.q_data is None or self.I_data is None:
            return
            
        # Make sure filtered indices are available
        if self.filtered_indices is None:
            self.filtered_indices = np.arange(len(self.q_data))
            
        # Get filtered data
        q_filtered = self.q_data[self.filtered_indices]
        I_filtered = self.I_data[self.filtered_indices]
        
        # Apply corrections to all data for visualization
        I_corrected_all = (self.I_data - self.bg_value) / self.norm_factor
        I_corrected_filtered = (I_filtered - self.bg_value) / self.norm_factor
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Set titles and labels
        self.ax1.set_title("SAXS Data")
        self.ax1.set_xlabel("q (Å⁻¹)")
        self.ax1.set_ylabel("I(q)")
        self.ax1.grid(True)
        
        self.ax2.set_title("Guinier Plot: ln(I) vs q²")
        self.ax2.set_xlabel("q² (Å⁻²)")
        self.ax2.set_ylabel("ln(I)")
        self.ax2.grid(True)
        
        self.ax3.set_title("Residuals")
        self.ax3.set_xlabel("q² (Å⁻²)")
        self.ax3.set_ylabel("ln(I) - fit")
        self.ax3.grid(True)
        self.ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot original data with error bars if available
        if hasattr(self, 'dI_data') and self.dI_data is not None:
            # All data with error bars
            dI_corrected = self.dI_data / self.norm_factor
            self.ax1.errorbar(self.q_data, I_corrected_all, yerr=dI_corrected, 
                             fmt='o', markersize=2, elinewidth=0.5, capsize=1, alpha=0.3,
                             label='All Data')
                             
            # Filtered data with error bars (points that meet SNR threshold)
            dI_filtered = self.dI_data[self.filtered_indices] / self.norm_factor
            self.ax1.errorbar(q_filtered, I_corrected_filtered, yerr=dI_filtered, 
                             fmt='o', markersize=4, elinewidth=1, capsize=2,
                             label=f'Filtered Data (SNR ≥ {self.snr_threshold})')
        else:
            # No error data available
            self.ax1.plot(self.q_data, I_corrected_all, 'o', markersize=2, alpha=0.3, label='All Data')
            self.ax1.plot(q_filtered, I_corrected_filtered, 'o', markersize=4, label='Filtered Data')
        
        # Highlight fit range
        q_range = q_filtered[self.q_min_idx:self.q_max_idx+1]
        I_range = I_corrected_filtered[self.q_min_idx:self.q_max_idx+1]
        
        if hasattr(self, 'dI_data') and self.dI_data is not None:
            dI_range = self.dI_data[self.filtered_indices][self.q_min_idx:self.q_max_idx+1] / self.norm_factor
            self.ax1.errorbar(q_range, I_range, yerr=dI_range, 
                             fmt='o', markersize=6, color='red', elinewidth=1, capsize=2,
                             label='Fit Range')
        else:
            self.ax1.plot(q_range, I_range, 'o', markersize=6, color='red', label='Fit Range')
        
        # Set y-scale to log for the SAXS plot
        self.ax1.set_yscale('log')
        self.ax1.set_xscale('log')  # Often SAXS is plotted with log scales
        
        # Add q·Rg = 1.3 line if Rg is available
        if self.Rg is not None:
            q_limit = 1.3 / self.Rg
            self.ax1.axvline(x=q_limit, color='red', linestyle='--', alpha=0.5, 
                            label=f'q·Rg = 1.3 (q = {q_limit:.4f})')
        
        # Plot Guinier data
        q_sq_all = self.q_data**2
        ln_I_all = np.log(I_corrected_all)
        valid_idx_all = np.isfinite(ln_I_all)
        
        q_sq_filtered = q_filtered**2
        ln_I_filtered = np.log(I_corrected_filtered)
        valid_idx_filtered = np.isfinite(ln_I_filtered)
        
        if np.any(valid_idx_all):
            # Plot all valid points
            self.ax2.plot(q_sq_all[valid_idx_all], ln_I_all[valid_idx_all], 
                         'o', markersize=2, alpha=0.3, label='All Data')
            
            # Plot filtered points
            if np.any(valid_idx_filtered):
                # If we have error data, calculate error in ln(I)
                if hasattr(self, 'dI_data') and self.dI_data is not None:
                    # Error propagation for ln(I): d(ln(I)) = dI/I
                    dln_I = self.dI_data[self.filtered_indices][valid_idx_filtered] / (
                        I_corrected_filtered[valid_idx_filtered] * self.norm_factor)
                    self.ax2.errorbar(q_sq_filtered[valid_idx_filtered], ln_I_filtered[valid_idx_filtered], 
                                     yerr=dln_I, fmt='o', markersize=4, elinewidth=1, capsize=2,
                                     label='Filtered Data')
                else:
                    self.ax2.plot(q_sq_filtered[valid_idx_filtered], ln_I_filtered[valid_idx_filtered], 
                                 'o', markersize=4, label='Filtered Data')
            
            # Highlight fit range
            q_sq_range = q_sq_filtered[self.q_min_idx:self.q_max_idx+1]
            ln_I_range = ln_I_filtered[self.q_min_idx:self.q_max_idx+1]
            valid_range_idx = np.isfinite(ln_I_range)
            
            if np.any(valid_range_idx):
                if hasattr(self, 'dI_data') and self.dI_data is not None:
                    # Calculate errors for fit range
                    dln_I_range = self.dI_data[self.filtered_indices][self.q_min_idx:self.q_max_idx+1][valid_range_idx] / (
                        I_range[valid_range_idx] * self.norm_factor)
                    self.ax2.errorbar(q_sq_range[valid_range_idx], ln_I_range[valid_range_idx], 
                                     yerr=dln_I_range, fmt='o', markersize=6, color='red', 
                                     elinewidth=1, capsize=2, label='Fit Range')
                else:
                    self.ax2.plot(q_sq_range[valid_range_idx], ln_I_range[valid_range_idx], 
                                 'o', markersize=6, color='red', label='Fit Range')
            
            # Show fit line if available
            if show_fit and fit_params is not None:
                intercept, slope = fit_params
                # Add fit details to plot title
                self.ax2.set_title(f"Guinier Plot: ln(I) vs q² [ln(I₀)={intercept:.2f}, Rg={self.Rg:.2f} Å]")
                
                # Draw fit line over the entire q range for visualization
                fit_line = intercept + slope * q_sq_range
                self.ax2.plot(q_sq_range, fit_line, '-', color='green', linewidth=2, 
                             label=f'Fit: ln(I) = {intercept:.2f} - ({-slope:.6f})q²')
                
                # Mark the q·Rg = 1.3 point
                q_sq_limit = (1.3 / self.Rg)**2
                self.ax2.axvline(x=q_sq_limit, color='red', linestyle='--', alpha=0.5, 
                               label=f'q·Rg = 1.3')
                
                # Also show Guinier curve on the original data
                if self.Rg is not None and self.I0 is not None:
                    guinier_curve = self.I0 * np.exp(-(q_range**2) * self.Rg**2 / 3)
                    self.ax1.plot(q_range, guinier_curve, '-', color='green', linewidth=2,
                                 label=f'Guinier Fit (Rg={self.Rg:.2f} Å)')
                    
                    # Plot residuals
                    if np.any(valid_range_idx):
                        residuals = ln_I_range[valid_range_idx] - (intercept + slope * q_sq_range[valid_range_idx])
                        self.ax3.plot(q_sq_range[valid_range_idx], residuals, 'o', color='blue')
                        
                        # Add fit quality metrics to residual plot
                        if self.r_squared is not None and self.chi_squared is not None:
                            quality_text = f"R² = {self.r_squared:.4f}, χ²ᵣₑₙ = {self.chi_squared:.4f}"
                            self.ax3.text(0.02, 0.95, quality_text, transform=self.ax3.transAxes,
                                         verticalalignment='top', horizontalalignment='left',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                        
                        # Calculate and show residual statistics
                        mean_residual = np.mean(residuals)
                        std_residual = np.std(residuals)
                        self.ax3.axhline(y=mean_residual, color='r', linestyle='--', 
                                        label=f'Mean: {mean_residual:.4f}')
                        self.ax3.axhline(y=mean_residual + std_residual, color='g', linestyle=':', 
                                        label=f'+1σ: {std_residual:.4f}')
                        self.ax3.axhline(y=mean_residual - std_residual, color='g', linestyle=':', 
                                        label=f'-1σ')
        
        # Add legends
        self.ax1.legend(loc='best')
        self.ax2.legend(loc='best')
        self.ax3.legend(loc='best')
        
        # Adjust layout
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_results(self):
        if self.Rg is None or self.I0 is None:
            return
            
        # Enable text widget for editing
        self.result_text.config(state="normal")
        
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Add results
        self.result_text.insert(tk.END, "Guinier Analysis Results:\n")
        self.result_text.insert(tk.END, "-----------------------\n")
        self.result_text.insert(tk.END, f"Radius of Gyration (Rg): {self.Rg:.2f} ± {self.Rg_error:.2f} Å\n")
        self.result_text.insert(tk.END, f"Zero-angle Intensity (I₀): {self.I0:.2e} ± {self.I0_error:.2e}\n")
        
        # Add validation metrics
        if self.r_squared is not None:
            r2_status = "Good" if self.r_squared > 0.99 else "Poor"
            self.result_text.insert(tk.END, f"R² (Goodness of fit): {self.r_squared:.4f} ({r2_status})\n")
            
        if self.chi_squared is not None:
            chi2_status = "Good" if 0.5 <= self.chi_squared <= 1.5 else "Poor"
            self.result_text.insert(tk.END, f"χ²ᵣₑₙ (Reduced chi-squared): {self.chi_squared:.4f} ({chi2_status})\n")
        
        # Get filtered q values
        if self.filtered_indices is not None and len(self.filtered_indices) > 0:
            q_filtered = self.q_data[self.filtered_indices]
            q_range = q_filtered[self.q_min_idx:self.q_max_idx+1]
            max_q_rg = q_range[-1] * self.Rg
            q_rg_status = "Valid" if max_q_rg <= 1.3 else "Exceeds limit"
            self.result_text.insert(tk.END, f"Fitting Range: q·Rg ≤ {max_q_rg:.2f} ({q_rg_status})\n")
            self.result_text.insert(tk.END, f"q range: {q_range[0]:.4f} - {q_range[-1]:.4f} Å⁻¹\n")
            
        # Add physical reasonability check
        if self.Rg > 0:
            # Calculate expected Rg based on simple geometric models
            # For example, for globular proteins, Rg ≈ 0.77 * (MW)^(1/3) nm, where MW is in kDa
            # This is just a rough guideline - users should compare with their specific systems
            physical_note = "Note: For validation, compare Rg with expected values for your system.\n"
            physical_note += "- Globular proteins: Rg ≈ 0.77 * (MW in kDa)^(1/3) nm\n"
            physical_note += "- Extended proteins: Rg may be 1.5-2x larger than globular\n"
            physical_note += "- Verify results with literature or known structures\n"
            self.result_text.insert(tk.END, f"\nPhysical Reasonability:\n{physical_note}\n")
        
        self.result_text.insert(tk.END, "-----------------------\n")
        
        # Disable text widget again
        self.result_text.config(state="disabled")
    
    def save_results(self):
        if self.Rg is None or self.I0 is None:
            messagebox.showwarning("Warning", "No fitting results to save")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Fitting Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt")]
        )
        
        if not filename:
            return
            
        try:
            # Create a DataFrame with the results
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
            
            # Get filtered q values
            if self.filtered_indices is not None and len(self.filtered_indices) > 0:
                q_filtered = self.q_data[self.filtered_indices]
                q_range = q_filtered[self.q_min_idx:self.q_max_idx+1]
                max_q_rg = q_range[-1] * self.Rg
                
                parameters.append("Fitting Range (q·Rg max)")
                values.append(f"{max_q_rg:.4f}")
                errors.append("")
                
                parameters.append("q min (Å⁻¹)")
                values.append(f"{q_range[0]:.6f}")
                errors.append("")
                
                parameters.append("q max (Å⁻¹)")
                values.append(f"{q_range[-1]:.6f}")
                errors.append("")
            
            parameters.extend(["Background Value", "Normalization Factor", "SNR Threshold"])
            values.extend([f"{self.bg_value:.4f}", f"{self.norm_factor:.4f}", f"{self.snr_threshold:.1f}"])
            errors.extend(["", "", ""])
            
            results = {
                "Parameter": parameters,
                "Value": values,
                "Error": errors
            }
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(filename, index=False)
            
            # Also save the fitted data points
            data_filename = os.path.splitext(filename)[0] + "_data.csv"
            
            # Get filtered data
            if self.filtered_indices is not None and len(self.filtered_indices) > 0:
                q_filtered = self.q_data[self.filtered_indices]
                I_filtered = self.I_data[self.filtered_indices]
                
                # Get fit range data
                q_range = q_filtered[self.q_min_idx:self.q_max_idx+1]
                I_range = I_filtered[self.q_min_idx:self.q_max_idx+1]
                I_corrected = (I_range - self.bg_value) / self.norm_factor
                
                # Calculate the Guinier fit
                guinier_fit = self.I0 * np.exp(-(q_range**2) * self.Rg**2 / 3)
                
                # Create a DataFrame with the data
                data = {
                    "q (Å⁻¹)": q_range,
                    "I_raw": I_range,
                    "I_corrected": I_corrected,
                    "ln(I)": np.log(I_corrected),
                    "q² (Å⁻²)": q_range**2,
                    "Guinier_fit": guinier_fit,
                    "q·Rg": q_range * self.Rg
                }
                
                if hasattr(self, 'dI_data') and self.dI_data is not None and self.filtered_indices is not None:
                    dI_filtered = self.dI_data[self.filtered_indices]
                    dI_range = dI_filtered[self.q_min_idx:self.q_max_idx+1] / self.norm_factor
                    data["dI"] = dI_range
                    data["SNR"] = I_corrected / dI_range
                    
                data_df = pd.DataFrame(data)
                data_df.to_csv(data_filename, index=False)
            
                # Save a report with plots
                report_filename = os.path.splitext(filename)[0] + "_report.pdf"
                try:
                    self.fig.savefig(report_filename)
                    messagebox.showinfo("Success", 
                                      f"Results saved to {filename}\nData saved to {data_filename}\nPlots saved to {report_filename}")
                except Exception as e:
                    messagebox.showinfo("Success", 
                                     f"Results saved to {filename}\nData saved to {data_filename}\nCould not save plots: {str(e)}")
            else:
                messagebox.showinfo("Success", f"Results saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GuinierAnalysisApp(root)
    root.mainloop() 
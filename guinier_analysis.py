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
        self.bg_value = 0.0
        self.norm_factor = 1.0
        self.q_min_idx = 0
        self.q_max_idx = -1
        self.Rg = None
        self.Rg_error = None
        self.I0 = None
        self.I0_error = None
        
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
        
        ttk.Label(load_frame, text="Data Format: Two columns (q, I)").pack(padx=5, pady=2)
        
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
        
        ttk.Button(fit_frame, text="Perform Guinier Fit", command=self.perform_fit).pack(padx=5, pady=5)
        
        # Results
        self.result_text = tk.Text(control_frame, height=10, width=40, state="disabled")
        self.result_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Right panel (plots)
        plot_frame = ttk.LabelFrame(main_frame, text="Plots")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create plots
        self.fig = Figure(figsize=(10, 8))
        
        # Raw data plot
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("SAXS Data")
        self.ax1.set_xlabel("q (Å⁻¹)")
        self.ax1.set_ylabel("I(q)")
        self.ax1.grid(True)
        
        # Guinier plot
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Guinier Plot: ln(I) vs q²")
        self.ax2.set_xlabel("q² (Å⁻²)")
        self.ax2.set_ylabel("ln(I)")
        self.ax2.grid(True)
        
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
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            # Try to load data with pandas
            data = pd.read_csv(filename, sep=None, engine='python', header=None)
            
            # If more than 2 columns, use the first two
            if data.shape[1] >= 2:
                self.q_data = data.iloc[:, 0].values
                self.I_data = data.iloc[:, 1].values
            else:
                raise ValueError("Data file must have at least 2 columns (q and I)")
                
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
            
            self.update_plots()
            messagebox.showinfo("Success", "Applied background subtraction and normalization")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid background or normalization values")
    
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
            # Use full range for initial estimate
            q_sq = self.q_data**2
            I_corrected = (self.I_data - self.bg_value) / self.norm_factor
            ln_I = np.log(I_corrected)
            
            # Fit to get rough Rg estimate
            valid_idx = np.isfinite(ln_I)
            if not np.any(valid_idx):
                raise ValueError("No valid points for fitting")
                
            params, _ = np.polyfit(q_sq[valid_idx], ln_I[valid_idx], 1, cov=True)
            slope = params[0]
            initial_Rg = np.sqrt(-3 * slope)
            
            # Find range where q*Rg <= 1.3
            q_Rg = self.q_data * initial_Rg
            valid_pts = np.where(q_Rg <= 1.3)[0]
            
            if len(valid_pts) > 0:
                self.q_min_idx = 0
                self.q_max_idx = valid_pts[-1]
                
                self.q_min_var.set(str(self.q_min_idx))
                self.q_max_var.set(str(self.q_max_idx))
                
                self.update_plots()
                messagebox.showinfo("Success", f"Auto set range where q·Rg ≤ 1.3 (Estimated Rg: {initial_Rg:.2f} Å)")
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
            # Get data in range
            q_range = self.q_data[self.q_min_idx:self.q_max_idx+1]
            I_range = self.I_data[self.q_min_idx:self.q_max_idx+1]
            
            # Apply corrections
            I_corrected = (I_range - self.bg_value) / self.norm_factor
            
            # Create Guinier plot data
            q_sq = q_range**2
            ln_I = np.log(I_corrected)
            
            # Check for valid data
            valid_idx = np.isfinite(ln_I)
            if not np.any(valid_idx):
                raise ValueError("No valid points for fitting")
                
            # Linear fit on ln(I) vs q^2
            params, pcov = np.polyfit(q_sq[valid_idx], ln_I[valid_idx], 1, cov=True)
            slope, intercept = params
            
            # Calculate Rg and I0
            self.Rg = np.sqrt(-3 * slope)
            self.I0 = np.exp(intercept)
            
            # Calculate errors
            slope_err, intercept_err = np.sqrt(np.diag(pcov))
            self.Rg_error = self.Rg * 0.5 * slope_err / abs(slope)
            self.I0_error = self.I0 * intercept_err
            
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
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Set titles and labels
        self.ax1.set_title("SAXS Data")
        self.ax1.set_xlabel("q (Å⁻¹)")
        self.ax1.set_ylabel("I(q)")
        self.ax1.grid(True)
        
        self.ax2.set_title("Guinier Plot: ln(I) vs q²")
        self.ax2.set_xlabel("q² (Å⁻²)")
        self.ax2.set_ylabel("ln(I)")
        self.ax2.grid(True)
        
        # Apply corrections
        I_corrected = (self.I_data - self.bg_value) / self.norm_factor
        
        # Plot original data
        self.ax1.plot(self.q_data, self.I_data, 'o', markersize=4, label='Raw Data')
        self.ax1.plot(self.q_data, I_corrected, 'x', markersize=4, label='Corrected Data')
        
        # Highlight fit range
        q_range = self.q_data[self.q_min_idx:self.q_max_idx+1]
        I_range = I_corrected[self.q_min_idx:self.q_max_idx+1]
        self.ax1.plot(q_range, I_range, 'o', markersize=6, color='red', label='Fit Range')
        
        # Plot Guinier data
        q_sq = self.q_data**2
        ln_I = np.log(I_corrected)
        valid_idx = np.isfinite(ln_I)
        
        if np.any(valid_idx):
            self.ax2.plot(q_sq[valid_idx], ln_I[valid_idx], 'o', markersize=4, label='ln(I) vs q²')
            
            # Highlight fit range
            q_sq_range = q_sq[self.q_min_idx:self.q_max_idx+1]
            ln_I_range = ln_I[self.q_min_idx:self.q_max_idx+1]
            valid_range_idx = np.isfinite(ln_I_range)
            
            if np.any(valid_range_idx):
                self.ax2.plot(q_sq_range[valid_range_idx], ln_I_range[valid_range_idx], 
                             'o', markersize=6, color='red', label='Fit Range')
            
            # Show fit line if available
            if show_fit and fit_params is not None:
                intercept, slope = fit_params
                fit_line = intercept + slope * q_sq_range
                self.ax2.plot(q_sq_range, fit_line, '-', color='green', linewidth=2, 
                             label=f'Fit: ln(I₀)={intercept:.2f}, -Rg²/3={slope:.6f}')
                
                # Also show Guinier curve on the original data
                if self.Rg is not None and self.I0 is not None:
                    guinier_curve = self.I0 * np.exp(-(q_range**2) * self.Rg**2 / 3)
                    self.ax1.plot(q_range, guinier_curve, '-', color='green', linewidth=2,
                                 label=f'Guinier Fit')
        
        self.ax1.legend()
        self.ax2.legend()
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
        self.result_text.insert(tk.END, f"Fitting Range: q·Rg ≤ {self.q_data[self.q_max_idx] * self.Rg:.2f}\n")
        self.result_text.insert(tk.END, "-----------------------\n")
        
        # Disable text widget again
        self.result_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = GuinierAnalysisApp(root)
    root.mainloop() 
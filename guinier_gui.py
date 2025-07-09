import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from guinier_core import GuinierAnalyzer


class GuinierAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Guinier Analysis for SAXS Data")
        self.root.geometry("1200x800")
        
        # Initialize core analyzer
        self.analyzer = GuinierAnalyzer()
        
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
        
        # Use core analyzer to load data
        result = self.analyzer.load_data(filename)
        
        if result['success']:
            # Update GUI variables
            self.q_min_var.set(str(self.analyzer.q_min_idx))
            self.q_max_var.set(str(self.analyzer.q_max_idx))
            self.bg_var.set(str(self.analyzer.bg_value))
            self.norm_var.set(str(self.analyzer.norm_factor))
            self.snr_var.set(str(self.analyzer.snr_threshold))
            
            # Update plots
            self.update_plots()
            messagebox.showinfo("Success", result['message'])
        else:
            messagebox.showerror("Error", result['message'])
    
    def apply_corrections(self):
        try:
            bg_value = float(self.bg_var.get())
            norm_factor = float(self.norm_var.get())
            snr_threshold = float(self.snr_var.get())
            
            # Use core analyzer to apply corrections
            result = self.analyzer.apply_corrections(bg_value, norm_factor, snr_threshold)
            
            if result['success']:
                # Update GUI variables
                self.q_min_var.set(str(self.analyzer.q_min_idx))
                self.q_max_var.set(str(self.analyzer.q_max_idx))
                
                self.update_plots()
                messagebox.showinfo("Success", result['message'])
            else:
                messagebox.showerror("Error", result['message'])
                
        except ValueError:
            messagebox.showerror("Error", "Invalid correction parameters")
    
    def set_fit_range(self):
        try:
            q_min_idx = int(self.q_min_var.get())
            q_max_idx = int(self.q_max_var.get())
            
            # Use core analyzer to set range
            result = self.analyzer.set_fit_range(q_min_idx, q_max_idx)
            
            if result['success']:
                self.update_plots()
                messagebox.showinfo("Success", result['message'])
            else:
                messagebox.showerror("Error", result['message'])
                
        except ValueError:
            messagebox.showerror("Error", "Invalid fit range parameters")
    
    def auto_range(self):
        # Use core analyzer to auto-set range
        result = self.analyzer.auto_range()
        
        if result['success']:
            # Update GUI variables
            self.q_min_var.set(str(result['q_min_idx']))
            self.q_max_var.set(str(result['q_max_idx']))
            
            self.update_plots()
            message = f"{result['message']} (Estimated Rg: {result['initial_Rg']:.2f} Å)"
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", result['message'])
    
    def perform_fit(self):
        use_robust = self.robust_var.get()
        
        # Use core analyzer to perform fit
        result = self.analyzer.perform_fit(use_robust)
        
        if result['success']:
            # Update plots with fit
            self.update_plots(show_fit=True)
            
            # Update results display
            self.update_results()
            
            message = result['message']
            if result.get('warning'):
                message += f"\n\n{result['warning']}"
            
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", result['message'])
    
    def update_plots(self, show_fit=False):
        # Get processed data from analyzer
        data = self.analyzer.get_processed_data()
        if not data:
            return
        
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
        
        # Plot original data
        self._plot_saxs_data(data)
        
        # Plot Guinier data
        self._plot_guinier_data(data)
        
        # Show fit if available
        if show_fit:
            self._plot_fit_results(data)
        
        # Add legends
        self.ax1.legend(loc='best')
        self.ax2.legend(loc='best')
        if show_fit:
            self.ax3.legend(loc='best')
        
        # Adjust layout
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _plot_saxs_data(self, data):
        """Plot SAXS data on the first subplot"""
        # Plot original data with error bars if available
        if data.get('dI_original') is not None:
            # All data with error bars
            dI_corrected = data['dI_original'] / data['norm_factor']
            self.ax1.errorbar(data['q_original'], data['I_corrected_all'], yerr=dI_corrected, 
                             fmt='o', markersize=2, elinewidth=0.5, capsize=1, alpha=0.3,
                             label='All Data')
        else:
            # No error data available
            self.ax1.plot(data['q_original'], data['I_corrected_all'], 'o', markersize=2, alpha=0.3, label='All Data')
        
        # Plot filtered data if available
        if 'q_filtered' in data:
            if data.get('dI_filtered') is not None:
                dI_filtered = data['dI_filtered'] / data['norm_factor']
                self.ax1.errorbar(data['q_filtered'], data['I_corrected_filtered'], yerr=dI_filtered, 
                                 fmt='o', markersize=4, elinewidth=1, capsize=2,
                                 label=f'Filtered Data (SNR ≥ {data["snr_threshold"]})')
            else:
                self.ax1.plot(data['q_filtered'], data['I_corrected_filtered'], 'o', markersize=4, label='Filtered Data')
        
        # Highlight fit range
        if 'q_range' in data:
            if data.get('dI_range') is not None:
                dI_range = data['dI_range'] / data['norm_factor']
                self.ax1.errorbar(data['q_range'], data['I_corrected_range'], yerr=dI_range, 
                                 fmt='o', markersize=6, color='red', elinewidth=1, capsize=2,
                                 label='Fit Range')
            else:
                self.ax1.plot(data['q_range'], data['I_corrected_range'], 'o', markersize=6, color='red', label='Fit Range')
        
        # Set scales
        self.ax1.set_yscale('log')
        self.ax1.set_xscale('log')
        
        # Add q·Rg = 1.3 line if fit results available
        fit_results = self.analyzer.get_fit_results()
        if fit_results and fit_results['Rg']:
            q_limit = 1.3 / fit_results['Rg']
            self.ax1.axvline(x=q_limit, color='red', linestyle='--', alpha=0.5, 
                            label=f'q·Rg = 1.3 (q = {q_limit:.4f})')
    
    def _plot_guinier_data(self, data):
        """Plot Guinier data on the second subplot"""
        # Plot all valid points
        q_sq_all = data['q_original']**2
        ln_I_all = np.log(data['I_corrected_all'])
        valid_idx_all = np.isfinite(ln_I_all)
        
        if np.any(valid_idx_all):
            self.ax2.plot(q_sq_all[valid_idx_all], ln_I_all[valid_idx_all], 
                         'o', markersize=2, alpha=0.3, label='All Data')
        
        # Plot filtered points
        if 'q_filtered' in data:
            q_sq_filtered = data['q_filtered']**2
            ln_I_filtered = np.log(data['I_corrected_filtered'])
            valid_idx_filtered = np.isfinite(ln_I_filtered)
            
            if np.any(valid_idx_filtered):
                if data.get('dI_filtered') is not None:
                    # Error propagation for ln(I): d(ln(I)) = dI/I
                    dln_I = data['dI_filtered'][valid_idx_filtered] / (
                        data['I_corrected_filtered'][valid_idx_filtered] * data['norm_factor'])
                    self.ax2.errorbar(q_sq_filtered[valid_idx_filtered], ln_I_filtered[valid_idx_filtered], 
                                     yerr=dln_I, fmt='o', markersize=4, elinewidth=1, capsize=2,
                                     label='Filtered Data')
                else:
                    self.ax2.plot(q_sq_filtered[valid_idx_filtered], ln_I_filtered[valid_idx_filtered], 
                                 'o', markersize=4, label='Filtered Data')
        
        # Highlight fit range
        if 'q_range' in data:
            q_sq_range = data['q_range']**2
            ln_I_range = np.log(data['I_corrected_range'])
            valid_range_idx = np.isfinite(ln_I_range)
            
            if np.any(valid_range_idx):
                if data.get('dI_range') is not None:
                    dln_I_range = data['dI_range'][valid_range_idx] / (
                        data['I_corrected_range'][valid_range_idx] * data['norm_factor'])
                    self.ax2.errorbar(q_sq_range[valid_range_idx], ln_I_range[valid_range_idx], 
                                     yerr=dln_I_range, fmt='o', markersize=6, color='red', 
                                     elinewidth=1, capsize=2, label='Fit Range')
                else:
                    self.ax2.plot(q_sq_range[valid_range_idx], ln_I_range[valid_range_idx], 
                                 'o', markersize=6, color='red', label='Fit Range')
    
    def _plot_fit_results(self, data):
        """Plot fit results on all subplots"""
        fit_results = self.analyzer.get_fit_results()
        if not fit_results:
            return
        
        # Generate fit curve
        fit_curve = self.analyzer.generate_fit_curve()
        if not fit_curve:
            return
        
        # Update Guinier plot title with fit info
        self.ax2.set_title(f"Guinier Plot: ln(I) vs q² [ln(I₀)={fit_results['fit_intercept']:.2f}, Rg={fit_results['Rg']:.2f} Å]")
        
        # Draw fit line on Guinier plot
        self.ax2.plot(fit_curve['q_sq'], fit_curve['ln_I_fit'], '-', color='green', linewidth=2, 
                     label=f'Fit: ln(I) = {fit_results["fit_intercept"]:.2f} - ({-fit_results["fit_slope"]:.6f})q²')
        
        # Mark the q·Rg = 1.3 point
        q_sq_limit = (1.3 / fit_results['Rg'])**2
        self.ax2.axvline(x=q_sq_limit, color='red', linestyle='--', alpha=0.5, 
                       label=f'q·Rg = 1.3')
        
        # Show Guinier curve on the original data
        self.ax1.plot(fit_curve['q_values'], fit_curve['I_fit'], '-', color='green', linewidth=2,
                     label=f'Guinier Fit (Rg={fit_results["Rg"]:.2f} Å)')
        
        # Plot residuals
        if 'q_range' in data:
            ln_I_range = np.log(data['I_corrected_range'])
            valid_range_idx = np.isfinite(ln_I_range)
            
            if np.any(valid_range_idx):
                residuals = ln_I_range[valid_range_idx] - fit_curve['ln_I_fit'][valid_range_idx]
                self.ax3.plot(fit_curve['q_sq'][valid_range_idx], residuals, 'o', color='blue')
                
                # Add fit quality metrics
                if fit_results['r_squared'] is not None and fit_results['chi_squared'] is not None:
                    quality_text = f"R² = {fit_results['r_squared']:.4f}, χ²ᵣₑₙ = {fit_results['chi_squared']:.4f}"
                    self.ax3.text(0.02, 0.95, quality_text, transform=self.ax3.transAxes,
                                 verticalalignment='top', horizontalalignment='left',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Show residual statistics
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                self.ax3.axhline(y=mean_residual, color='r', linestyle='--', 
                                label=f'Mean: {mean_residual:.4f}')
                self.ax3.axhline(y=mean_residual + std_residual, color='g', linestyle=':', 
                                label=f'+1σ: {std_residual:.4f}')
                self.ax3.axhline(y=mean_residual - std_residual, color='g', linestyle=':', 
                                label=f'-1σ')
    
    def update_results(self):
        """Update the results text display"""
        fit_results = self.analyzer.get_fit_results()
        if not fit_results:
            return
        
        # Enable text widget for editing
        self.result_text.config(state="normal")
        
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Add results
        self.result_text.insert(tk.END, "Guinier Analysis Results:\n")
        self.result_text.insert(tk.END, "-----------------------\n")
        self.result_text.insert(tk.END, f"Radius of Gyration (Rg): {fit_results['Rg']:.2f} ± {fit_results['Rg_error']:.2f} Å\n")
        self.result_text.insert(tk.END, f"Zero-angle Intensity (I₀): {fit_results['I0']:.2e} ± {fit_results['I0_error']:.2e}\n")
        
        # Add validation metrics
        if fit_results['r_squared'] is not None:
            r2_status = "Good" if fit_results['r_squared'] > 0.99 else "Poor"
            self.result_text.insert(tk.END, f"R² (Goodness of fit): {fit_results['r_squared']:.4f} ({r2_status})\n")
        
        if fit_results['chi_squared'] is not None:
            chi2_status = "Good" if 0.5 <= fit_results['chi_squared'] <= 1.5 else "Poor"
            self.result_text.insert(tk.END, f"χ²ᵣₑₙ (Reduced chi-squared): {fit_results['chi_squared']:.4f} ({chi2_status})\n")
        
        # Add fitting range info
        if fit_results['max_q_rg'] is not None:
            q_rg_status = "Valid" if fit_results['max_q_rg'] <= 1.3 else "Exceeds limit"
            self.result_text.insert(tk.END, f"Fitting Range: q·Rg ≤ {fit_results['max_q_rg']:.2f} ({q_rg_status})\n")
        
        data = self.analyzer.get_processed_data()
        if data and 'q_range' in data:
            self.result_text.insert(tk.END, f"q range: {data['q_range'][0]:.4f} - {data['q_range'][-1]:.4f} Å⁻¹\n")
        
        # Add physical reasonability check
        if fit_results['Rg'] > 0:
            physical_note = "Note: For validation, compare Rg with expected values for your system.\n"
            physical_note += "- Globular proteins: Rg ≈ 0.77 * (MW in kDa)^(1/3) nm\n"
            physical_note += "- Extended proteins: Rg may be 1.5-2x larger than globular\n"
            physical_note += "- Verify results with literature or known structures\n"
            self.result_text.insert(tk.END, f"\nPhysical Reasonability:\n{physical_note}\n")
        
        self.result_text.insert(tk.END, "-----------------------\n")
        
        # Disable text widget again
        self.result_text.config(state="disabled")
    
    def save_results(self):
        """Save results using the core analyzer"""
        fit_results = self.analyzer.get_fit_results()
        if not fit_results:
            messagebox.showwarning("Warning", "No fitting results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Fitting Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt")]
        )
        
        if not filename:
            return
        
        # Use core analyzer to save results
        result = self.analyzer.save_results(filename)
        
        if result['success']:
            # Try to save plot
            try:
                report_filename = os.path.splitext(filename)[0] + "_report.pdf"
                self.fig.savefig(report_filename)
                message = f"{result['message']}\nData saved to {result.get('data_file', 'N/A')}\nPlots saved to {report_filename}"
            except Exception as e:
                message = f"{result['message']}\nData saved to {result.get('data_file', 'N/A')}\nCould not save plots: {str(e)}"
            
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", result['message'])


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = GuinierAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 
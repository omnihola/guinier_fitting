#!/usr/bin/env python3
"""
Example usage of the Guinier Analysis Core Module

This script demonstrates how to use the modular Guinier analysis functionality
without the GUI, which is useful for batch processing or integration into
other analysis pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from guinier_core import GuinierAnalyzer


def analyze_saxs_data(filename, bg_value=0.0, norm_factor=1.0, snr_threshold=3.0):
    """
    Perform complete Guinier analysis on SAXS data.
    
    Parameters:
    -----------
    filename : str
        Path to the SAXS data file
    bg_value : float
        Background value to subtract
    norm_factor : float
        Normalization factor
    snr_threshold : float
        Minimum signal-to-noise ratio threshold
        
    Returns:
    --------
    dict : Analysis results
    """
    
    # Initialize analyzer
    analyzer = GuinierAnalyzer()
    
    # Load data
    print(f"Loading data from {filename}...")
    load_result = analyzer.load_data(filename)
    
    if not load_result['success']:
        print(f"Error loading data: {load_result['message']}")
        return None
    
    print(f"Successfully loaded {load_result['n_points']} data points")
    
    # Apply corrections
    print("Applying corrections...")
    correction_result = analyzer.apply_corrections(bg_value, norm_factor, snr_threshold)
    
    if not correction_result['success']:
        print(f"Error applying corrections: {correction_result['message']}")
        return None
    
    print(f"Applied corrections: {correction_result['n_filtered']}/{correction_result['n_total']} points retained")
    
    # Auto-determine fitting range
    print("Auto-determining fitting range...")
    auto_range_result = analyzer.auto_range()
    
    if not auto_range_result['success']:
        print(f"Error auto-setting range: {auto_range_result['message']}")
        return None
    
    print(f"Auto-set range: indices {auto_range_result['q_min_idx']} to {auto_range_result['q_max_idx']}")
    print(f"Initial Rg estimate: {auto_range_result['initial_Rg']:.2f} Å")
    
    # Perform fit
    print("Performing Guinier fit...")
    fit_result = analyzer.perform_fit(use_robust=True)
    
    if not fit_result['success']:
        print(f"Error performing fit: {fit_result['message']}")
        return None
    
    print(f"Fit completed successfully!")
    if fit_result.get('warning'):
        print(f"Warning: {fit_result['warning']}")
    
    # Get results
    results = analyzer.get_fit_results()
    
    return {
        'analyzer': analyzer,
        'results': results,
        'load_result': load_result,
        'correction_result': correction_result,
        'auto_range_result': auto_range_result,
        'fit_result': fit_result
    }


def print_results(analysis_results):
    """Print analysis results in a formatted way."""
    
    if not analysis_results:
        print("No results to display")
        return
    
    results = analysis_results['results']
    
    print("\n" + "="*50)
    print("GUINIER ANALYSIS RESULTS")
    print("="*50)
    
    print(f"Radius of Gyration (Rg): {results['Rg']:.2f} ± {results['Rg_error']:.2f} Å")
    print(f"Zero-angle Intensity (I₀): {results['I0']:.2e} ± {results['I0_error']:.2e}")
    
    # Fit quality
    print(f"\nFit Quality:")
    print(f"  R² (Goodness of fit): {results['r_squared']:.4f}")
    r2_status = "Good" if results['r_squared'] > 0.99 else "Poor"
    print(f"  Status: {r2_status}")
    
    print(f"  χ²ᵣₑₙ (Reduced chi-squared): {results['chi_squared']:.4f}")
    chi2_status = "Good" if 0.5 <= results['chi_squared'] <= 1.5 else "Poor"
    print(f"  Status: {chi2_status}")
    
    # Validity check
    if results['max_q_rg'] is not None:
        print(f"\nValidity Check:")
        print(f"  Maximum q·Rg: {results['max_q_rg']:.2f}")
        validity_status = "Valid" if results['max_q_rg'] <= 1.3 else "Exceeds limit"
        print(f"  Status: {validity_status}")
    
    print("\n" + "="*50)


def plot_results(analysis_results, save_filename=None):
    """
    Create publication-quality plots of the analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from analyze_saxs_data function
    save_filename : str, optional
        If provided, save plots to this file
    """
    
    if not analysis_results:
        print("No results to plot")
        return
    
    analyzer = analysis_results['analyzer']
    results = analysis_results['results']
    
    # Get processed data
    data = analyzer.get_processed_data()
    if not data:
        print("No processed data available for plotting")
        return
    
    # Get fit curve
    fit_curve = analyzer.generate_fit_curve()
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: SAXS Data
    ax1.set_title("SAXS Data", fontsize=14, fontweight='bold')
    ax1.set_xlabel("q (Å⁻¹)")
    ax1.set_ylabel("I(q)")
    ax1.grid(True, alpha=0.3)
    
    # Plot all data
    if data.get('dI_original') is not None:
        dI_corrected = data['dI_original'] / data['norm_factor']
        ax1.errorbar(data['q_original'], data['I_corrected_all'], yerr=dI_corrected, 
                    fmt='o', markersize=2, alpha=0.3, label='All Data')
    else:
        ax1.plot(data['q_original'], data['I_corrected_all'], 'o', markersize=2, alpha=0.3, label='All Data')
    
    # Plot filtered data
    if 'q_filtered' in data:
        if data.get('dI_filtered') is not None:
            dI_filtered = data['dI_filtered'] / data['norm_factor']
            ax1.errorbar(data['q_filtered'], data['I_corrected_filtered'], yerr=dI_filtered, 
                        fmt='s', markersize=4, alpha=0.7, label='Filtered Data')
        else:
            ax1.plot(data['q_filtered'], data['I_corrected_filtered'], 's', markersize=4, alpha=0.7, label='Filtered Data')
    
    # Plot fit range
    if 'q_range' in data:
        if data.get('dI_range') is not None:
            dI_range = data['dI_range'] / data['norm_factor']
            ax1.errorbar(data['q_range'], data['I_corrected_range'], yerr=dI_range, 
                        fmt='o', markersize=6, color='red', label='Fit Range')
        else:
            ax1.plot(data['q_range'], data['I_corrected_range'], 'o', markersize=6, color='red', label='Fit Range')
    
    # Plot fit curve
    if fit_curve:
        ax1.plot(fit_curve['q_values'], fit_curve['I_fit'], '-', color='green', linewidth=2,
                label=f'Guinier Fit (Rg={results["Rg"]:.2f} Å)')
    
    # Add validity line
    if results['Rg']:
        q_limit = 1.3 / results['Rg']
        ax1.axvline(x=q_limit, color='red', linestyle='--', alpha=0.5, 
                   label=f'q·Rg = 1.3')
    
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    
    # Plot 2: Guinier Plot
    ax2.set_title(f"Guinier Plot: ln(I) vs q² [Rg={results['Rg']:.2f} Å]", fontsize=14, fontweight='bold')
    ax2.set_xlabel("q² (Å⁻²)")
    ax2.set_ylabel("ln(I)")
    ax2.grid(True, alpha=0.3)
    
    # Plot all data
    q_sq_all = data['q_original']**2
    ln_I_all = np.log(data['I_corrected_all'])
    valid_idx_all = np.isfinite(ln_I_all)
    
    if np.any(valid_idx_all):
        ax2.plot(q_sq_all[valid_idx_all], ln_I_all[valid_idx_all], 
                'o', markersize=2, alpha=0.3, label='All Data')
    
    # Plot filtered data
    if 'q_filtered' in data:
        q_sq_filtered = data['q_filtered']**2
        ln_I_filtered = np.log(data['I_corrected_filtered'])
        valid_idx_filtered = np.isfinite(ln_I_filtered)
        
        if np.any(valid_idx_filtered):
            ax2.plot(q_sq_filtered[valid_idx_filtered], ln_I_filtered[valid_idx_filtered], 
                    's', markersize=4, alpha=0.7, label='Filtered Data')
    
    # Plot fit range
    if 'q_range' in data:
        q_sq_range = data['q_range']**2
        ln_I_range = np.log(data['I_corrected_range'])
        valid_range_idx = np.isfinite(ln_I_range)
        
        if np.any(valid_range_idx):
            ax2.plot(q_sq_range[valid_range_idx], ln_I_range[valid_range_idx], 
                    'o', markersize=6, color='red', label='Fit Range')
    
    # Plot fit line
    if fit_curve:
        ax2.plot(fit_curve['q_sq'], fit_curve['ln_I_fit'], '-', color='green', linewidth=2, 
                label=f'Fit: ln(I) = {results["fit_intercept"]:.2f} - ({-results["fit_slope"]:.6f})q²')
        
        # Mark validity limit
        q_sq_limit = (1.3 / results['Rg'])**2
        ax2.axvline(x=q_sq_limit, color='red', linestyle='--', alpha=0.5, 
                   label=f'q·Rg = 1.3')
    
    ax2.legend()
    
    # Plot 3: Residuals
    ax3.set_title("Residuals", fontsize=14, fontweight='bold')
    ax3.set_xlabel("q² (Å⁻²)")
    ax3.set_ylabel("ln(I) - fit")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot residuals
    if fit_curve and 'q_range' in data:
        ln_I_range = np.log(data['I_corrected_range'])
        valid_range_idx = np.isfinite(ln_I_range)
        
        if np.any(valid_range_idx):
            residuals = ln_I_range[valid_range_idx] - fit_curve['ln_I_fit'][valid_range_idx]
            ax3.plot(fit_curve['q_sq'][valid_range_idx], residuals, 'o', color='blue')
            
            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax3.axhline(y=mean_residual, color='r', linestyle='--', 
                       label=f'Mean: {mean_residual:.4f}')
            ax3.axhline(y=mean_residual + std_residual, color='g', linestyle=':', 
                       label=f'+1σ: {std_residual:.4f}')
            ax3.axhline(y=mean_residual - std_residual, color='g', linestyle=':', 
                       label=f'-1σ')
            
            # Add fit quality text
            quality_text = f"R² = {results['r_squared']:.4f}, χ²ᵣₑₙ = {results['chi_squared']:.4f}"
            ax3.text(0.02, 0.95, quality_text, transform=ax3.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.legend()
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_filename}")
    
    plt.show()


def batch_analysis(filenames, output_dir=None):
    """
    Perform batch analysis on multiple SAXS files.
    
    Parameters:
    -----------
    filenames : list
        List of file paths to analyze
    output_dir : str, optional
        Directory to save results
    """
    
    import os
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    batch_results = []
    
    for filename in filenames:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(filename)}")
        print('='*60)
        
        # Analyze data
        analysis_result = analyze_saxs_data(filename)
        
        if analysis_result:
            # Print results
            print_results(analysis_result)
            
            # Save results if output directory specified
            if output_dir:
                base_name = os.path.splitext(os.path.basename(filename))[0]
                
                # Save numerical results
                result_file = os.path.join(output_dir, f"{base_name}_results.csv")
                save_result = analysis_result['analyzer'].save_results(result_file)
                if save_result['success']:
                    print(f"Results saved to {result_file}")
                
                # Save plot
                plot_file = os.path.join(output_dir, f"{base_name}_plot.png")
                plot_results(analysis_result, plot_file)
            
            batch_results.append({
                'filename': filename,
                'analysis': analysis_result,
                'success': True
            })
        else:
            print(f"Failed to analyze {filename}")
            batch_results.append({
                'filename': filename,
                'analysis': None,
                'success': False
            })
    
    return batch_results


def main():
    """Main function demonstrating usage."""
    
    print("Guinier Analysis Core Module - Example Usage")
    print("="*50)
    
    # Example 1: Single file analysis
    print("\nExample 1: Single file analysis")
    print("-" * 30)
    
    # You would replace this with your actual data file
    # analysis_result = analyze_saxs_data("your_data_file.grad")
    # 
    # if analysis_result:
    #     print_results(analysis_result)
    #     plot_results(analysis_result, "analysis_plot.png")
    
    print("To analyze a single file:")
    print("analysis_result = analyze_saxs_data('your_data_file.grad')")
    print("print_results(analysis_result)")
    print("plot_results(analysis_result, 'analysis_plot.png')")
    
    # Example 2: Batch analysis
    print("\nExample 2: Batch analysis")
    print("-" * 30)
    
    # You would replace this with your actual data files
    # filenames = ["file1.grad", "file2.grad", "file3.grad"]
    # batch_results = batch_analysis(filenames, output_dir="batch_results")
    
    print("To analyze multiple files:")
    print("filenames = ['file1.grad', 'file2.grad', 'file3.grad']")
    print("batch_results = batch_analysis(filenames, output_dir='batch_results')")
    
    # Example 3: Programmatic analysis with custom parameters
    print("\nExample 3: Custom analysis parameters")
    print("-" * 30)
    
    print("To use custom parameters:")
    print("analyzer = GuinierAnalyzer()")
    print("load_result = analyzer.load_data('your_file.grad')")
    print("correction_result = analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=5.0)")
    print("analyzer.set_fit_range(5, 50)  # Use specific index range")
    print("fit_result = analyzer.perform_fit(use_robust=False)")
    print("results = analyzer.get_fit_results()")
    
    print("\nFor more details, see the docstrings in guinier_core.py")


if __name__ == "__main__":
    main() 
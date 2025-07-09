# Guinier Analysis for SAXS Data

A modular Python application for analyzing Small Angle X-ray Scattering (SAXS) data using the Guinier approximation.

## Overview

This application provides both a graphical user interface (GUI) and a programmatic API for SAXS data analysis. The modular design allows for:

- **Interactive GUI analysis** for user-friendly data exploration
- **Programmatic batch processing** for high-throughput analysis
- **Easy integration** into existing analysis pipelines

### Features

- Load and visualize SAXS data (supports .grad files and standard q, I format)
- Background subtraction and normalization
- Signal-to-noise ratio (SNR) filtering
- Automated Guinier regime detection (q·Rg ≤ 1.3)
- Robust fitting algorithms with outlier rejection
- Comprehensive fit quality assessment
- Export results in CSV format with publication-quality plots

## Modular Architecture

The application is organized into three main modules:

- **`guinier_core.py`**: Core analysis functionality (data loading, processing, fitting)
- **`guinier_gui.py`**: Graphical user interface
- **`guinier_analysis.py`**: Main entry point (backward compatible)
- **`example_usage.py`**: Demonstrates programmatic usage

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application

Run the graphical interface:

```bash
python guinier_analysis.py
```

Or directly:

```bash
python guinier_gui.py
```

### Programmatic Usage

For batch processing or integration into analysis pipelines:

```python
from guinier_core import GuinierAnalyzer

# Initialize analyzer
analyzer = GuinierAnalyzer()

# Load data
result = analyzer.load_data("your_data.grad")

# Apply corrections
analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=3.0)

# Auto-determine fitting range
analyzer.auto_range()

# Perform fit
fit_result = analyzer.perform_fit(use_robust=True)

# Get results
results = analyzer.get_fit_results()
print(f"Rg = {results['Rg']:.2f} ± {results['Rg_error']:.2f} Å")
```

### Batch Processing

Process multiple files automatically:

```python
from example_usage import batch_analysis

filenames = ["file1.grad", "file2.grad", "file3.grad"]
batch_results = batch_analysis(filenames, output_dir="results")
```

## API Reference

### GuinierAnalyzer Class

The core analysis class with the following main methods:

- `load_data(filename)`: Load SAXS data from file
- `apply_corrections(bg_value, norm_factor, snr_threshold)`: Apply data corrections
- `auto_range(q_rg_limit=1.3)`: Automatically determine fitting range
- `set_fit_range(q_min_idx, q_max_idx)`: Manually set fitting range
- `perform_fit(use_robust=True)`: Perform Guinier fitting
- `get_fit_results()`: Get fitting results
- `get_processed_data()`: Get processed data arrays
- `save_results(filename)`: Save results to CSV file

### Data Formats

**Supported input formats:**
- `.grad` files (SAXS data with q, I, dI columns)
- CSV/text files with q, I columns (optional dI column)

**Output formats:**
- CSV files with fitting parameters and uncertainties
- CSV files with fitted data points
- PDF plots (when using GUI)

## Theoretical Background

The Guinier approximation describes the scattering intensity I(q) in the low-q region:

```
I(q) = I₀ · exp(-q²Rg²/3)
```

Where:
- **I₀** is the extrapolated intensity at q=0
- **Rg** is the radius of gyration of the scattering domains
- The approximation is valid in the region where **q·Rg ≤ 1.3**

By plotting ln(I) vs q², a linear relationship is expected in the Guinier regime:
- **Slope** = -Rg²/3
- **Intercept** = ln(I₀)

## Advanced Features

### Robust Fitting

The application includes robust fitting algorithms that reduce the impact of outliers:

- **Theil-Sen estimator**: Robust slope estimation
- **Huber regression**: Weighted robust fitting for data with error bars
- **Automatic outlier detection**: Based on residual analysis

### Quality Assessment

Comprehensive fit quality metrics:

- **R² (coefficient of determination)**: Measures goodness of fit
- **χ²ᵣₑₙ (reduced chi-squared)**: Measures fit quality relative to expected variance
- **Residual analysis**: Statistical analysis of fit residuals
- **Physical validity checks**: Ensures q·Rg ≤ 1.3 criterion

### SNR Filtering

Automatic data filtering based on signal-to-noise ratio:

- Filters out low-quality data points
- Preserves high-quality data for reliable fitting
- Configurable SNR threshold

## Example Workflows

### 1. Interactive Analysis

```python
# Load GUI for interactive analysis
python guinier_analysis.py

# 1. Load your SAXS data file
# 2. Adjust background and normalization if needed
# 3. Use "Auto Range" to find optimal fitting range
# 4. Perform Guinier fit
# 5. Save results and plots
```

### 2. Automated Batch Processing

```python
import os
from example_usage import batch_analysis

# Process all .grad files in a directory
data_dir = "saxs_data"
filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.grad')]

# Batch process with automatic output
results = batch_analysis(filenames, output_dir="batch_results")

# Analyze success rate
success_rate = sum(1 for r in results if r['success']) / len(results)
print(f"Success rate: {success_rate:.1%}")
```

### 3. Custom Analysis Pipeline

```python
from guinier_core import GuinierAnalyzer
import numpy as np

def custom_analysis(filename):
    analyzer = GuinierAnalyzer()
    
    # Load and process data
    analyzer.load_data(filename)
    analyzer.apply_corrections(bg_value=0.05, snr_threshold=5.0)
    
    # Use custom fitting range
    analyzer.set_fit_range(10, 100)
    
    # Perform fit with custom parameters
    fit_result = analyzer.perform_fit(use_robust=False)
    
    if fit_result['success']:
        results = analyzer.get_fit_results()
        
        # Custom validation
        if results['r_squared'] > 0.995 and results['max_q_rg'] <= 1.3:
            return results
    
    return None
```

## Notes

- **Data quality**: Ensure data is properly background-subtracted and normalized
- **Guinier regime**: The approximation is only valid for dilute, non-interacting systems
- **Validity condition**: Always verify that q·Rg ≤ 1.3 for reliable results
- **Error analysis**: Use error bars when available for more accurate uncertainty estimates

## Physical Validation Guidelines

For **globular proteins**:
- Rg ≈ 0.77 × (MW in kDa)^(1/3) nm

For **extended proteins**:
- Rg may be 1.5-2× larger than globular proteins

Always compare results with:
- Literature values for similar systems
- Known structural data
- Expected molecular weight relationships

## Troubleshooting

**Common issues:**

1. **Import errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
2. **Data loading fails**: Check file format and ensure q, I columns are present
3. **Fitting fails**: Verify data quality and try adjusting SNR threshold
4. **Invalid Rg**: Check that data contains a valid Guinier regime

**Getting help:**

- Check the example scripts for usage patterns
- Review docstrings in `guinier_core.py` for detailed API documentation
- Ensure your data follows the expected format conventions

## Requirements

- Python 3.6+
- numpy
- matplotlib
- scipy
- pandas
- tkinter (for GUI)
- scikit-learn (optional, for robust fitting)

## License

This project is provided as-is for research and educational purposes. 
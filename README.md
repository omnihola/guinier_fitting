# Guinier Analysis for SAXS Data

A modular Python application for Small-Angle X-ray Scattering (SAXS) Guinier analysis, featuring traditional and machine learning-based fitting algorithms.

## Features

- **Multiple Fitting Algorithms**: Traditional least squares, robust methods, and scikit-learn implementations
- **Interactive GUI**: User-friendly interface with real-time plotting and modern icon
- **Data Processing**: Background subtraction, normalization, and SNR filtering
- **Physical Validation**: Automatic q·Rg range checking and quality assessment
- **Cross-validation**: Model validation and comparison capabilities
- **Export Functionality**: Save results and fitted data to CSV files

## Installation

1. Clone or download the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Dependencies
- `numpy` - Numerical computations
- `scipy` - Scientific computing and optimization
- `pandas` - Data manipulation and analysis
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Machine learning algorithms (optional but recommended)
- `PySide6` - Modern GUI framework (recommended)
- `tkinter` - Classic GUI framework (fallback, usually included with Python)

## Quick Start

### GUI Application
```bash
python guinier_analysis.py
```

The application automatically detects and uses the best available GUI framework:
- **PySide6 Interface** (recommended): Modern, professional interface with enhanced styling
- **Tkinter Interface** (fallback): Classic interface for compatibility

For the best experience, install PySide6:
```bash
pip install PySide6
```

### Programmatic Usage
```python
from guinier_core import GuinierAnalyzer

# Initialize analyzer
analyzer = GuinierAnalyzer()

# Load SAXS data
analyzer.load_data('your_data.grad')

# Apply corrections
analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0)

# Set fitting range
analyzer.auto_range()  # Automatic range based on q·Rg ≤ 1.3

# Perform fitting
result = analyzer.perform_fit()  # Traditional method
sklearn_result = analyzer.fit_with_sklearn('huber')  # Sklearn method

# Compare methods
comparison = analyzer.compare_methods()
```

## Project Structure

```
guinier/
├── guinier_analysis.py       # Main entry point (auto-detects GUI)
├── guinier_core.py           # Core analysis engine
├── guinier_gui.py            # Classic tkinter GUI
├── guinier_gui_pyside6.py    # Modern PySide6 GUI (recommended)
├── guinier_icon.png          # Application icon (PNG format)
├── guinier_icon.ico          # Application icon (ICO format)
├── requirements.txt          # Dependencies
├── README.md                # This file
├── examples/                # Usage examples
│   └── example_usage.py
└── tests/                   # Test files
    └── test_gui_sklearn.py
```

## Algorithms Available

### Traditional Methods
- **traditional**: Standard least squares fitting using `numpy.polyfit`
- **traditional_robust**: Robust fitting using Theil-Sen and Huber methods

### Machine Learning Methods (sklearn)
- **linear**: Linear regression
- **huber**: Huber regression (robust to outliers)
- **ridge**: Ridge regression (L2 regularization)
- **theilsen**: Theil-Sen regression (very robust)
- **ransac**: RANSAC regression (random sample consensus)

### Algorithm Selection Guide
- **For clean data**: Use `traditional` or `linear`
- **For noisy data**: Use `huber` or `traditional_robust`
- **For data with outliers**: Use `theilsen` or `ransac`
- **For overfitting concerns**: Use `ridge`

## Core Classes

### GuinierAnalyzer
Main analysis class with comprehensive functionality:

**Key Methods:**
- `load_data(filename)`: Load SAXS data from various formats
- `apply_corrections(bg_value, norm_factor, snr_threshold)`: Process raw data
- `auto_range(q_rg_limit=1.3)`: Automatically set Guinier range
- `perform_fit(use_robust=True)`: Traditional Guinier fitting
- `fit_with_sklearn(algorithm, cross_validate=False)`: ML-based fitting
- `compare_methods()`: Compare all available methods
- `save_results(filename)`: Export results to CSV

**Results Properties:**
- `Rg`: Radius of gyration (Å)
- `I0`: Zero-angle intensity
- `r_squared`: Coefficient of determination
- `chi_squared`: Reduced chi-squared
- `sklearn_models`: Dictionary of fitted ML models

## GUI Features

### Interface Options
- **PySide6 Interface** (recommended): Modern, professional styling with enhanced user experience
- **Tkinter Interface** (fallback): Classic interface for compatibility

### Data Loading
- Support for `.grad`, `.txt`, and `.csv` formats
- Automatic format detection
- Data validation and error handling
- Progress indicators for long operations

### Processing Controls
- Background subtraction
- Normalization factor adjustment
- SNR threshold filtering
- Manual and automatic range selection
- Real-time parameter updates

### Fitting Options
- Algorithm selection dropdown
- Cross-validation toggle
- Robust fitting option
- Method comparison tool
- Best model recommendations

### Visualization
- Raw SAXS data plot with error bars
- Guinier plot (ln(I) vs q²) with fit visualization
- Residuals analysis with statistics
- Real-time updates and interactive plots
- High-quality plot export (PDF)

## Data Formats

### Supported Input Formats
1. **GRAD files** (`.grad`): Standard SAXS analysis format
2. **Text files** (`.txt`, `.csv`): Column-separated data
   - Column 1: q values (Å⁻¹)
   - Column 2: I(q) values
   - Column 3: dI (errors) - optional

### Output Formats
- **Results CSV**: Fitting parameters and quality metrics
- **Data CSV**: Processed data points and fitted curves

## Physical Validation

The application automatically validates Guinier fitting according to established principles:
- **q·Rg limit**: Ensures q·Rg ≤ 1.3 for valid approximation
- **Positive intensity**: Validates I(q) > 0
- **Negative slope**: Ensures proper Guinier behavior
- **Error propagation**: Handles uncertainties in fitting

## Cross-Validation

When enabled, the application performs k-fold cross-validation:
- Evaluates model stability and generalization
- Provides mean and standard deviation of scores
- Helps in algorithm selection
- Detects overfitting

## Error Handling

The application includes comprehensive error handling:
- Invalid data format detection
- Numerical computation errors
- File I/O errors
- Algorithm convergence issues

## Examples

See `examples/example_usage.py` for detailed usage examples including:
- Basic analysis workflow
- Advanced parameter tuning
- Method comparison
- Results interpretation

## Testing

Run the test suite:
```bash
python tests/test_gui_sklearn.py
```

## Troubleshooting

### Common Issues
1. **Scikit-learn not found**: Install with `pip install scikit-learn`
2. **PySide6 not found**: Install with `pip install PySide6` for the modern interface
3. **Fitting fails**: Check data quality and range selection
4. **GUI not starting**: Ensure either PySide6 or tkinter is available

### GUI Selection
The application automatically selects the best available GUI framework:
1. First tries PySide6 (modern interface)
2. Falls back to tkinter (classic interface)
3. If neither is available, displays installation instructions

### Additional Tips
- **Poor fit quality**: Try different algorithms or adjust range
- **GUI not responding**: Restart the application or check system requirements

### Performance Tips
- Use automatic range selection for optimal results
- Enable cross-validation for method comparison
- Choose robust algorithms for noisy data
- Validate q·Rg limits for physical accuracy

## Contributing

This is a focused analysis tool. For enhancements:
1. Follow the modular architecture
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

## License

This project is provided as-is for scientific and educational use. 
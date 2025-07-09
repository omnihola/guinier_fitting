# Guinier Analysis for SAXS Data

A comprehensive Python toolkit for Small Angle X-ray Scattering (SAXS) Guinier analysis with both traditional and machine learning approaches.

## Features

### Core Functionality
- **Multiple Data Format Support**: .grad files, standard text files with q, I columns
- **Data Processing**: Background subtraction, normalization, SNR filtering
- **Intelligent Fitting Range**: Manual selection and automatic range detection (q·Rg ≤ 1.3)
- **Robust Fitting Algorithms**: Traditional least squares, robust regression, and scikit-learn methods
- **Quality Assessment**: R², χ², residual analysis, and physical validation
- **Results Export**: Comprehensive results with uncertainty estimates

### Machine Learning Integration
- **Multiple Regression Algorithms**: Linear, Huber, RANSAC, Theil-Sen, Ridge, Lasso
- **Cross-Validation**: Model stability assessment
- **Hyperparameter Optimization**: Automatic parameter tuning
- **Model Comparison**: Side-by-side algorithm comparison
- **Pipeline Support**: Preprocessing and fitting in unified workflows

### User Interfaces
- **Interactive GUI**: User-friendly interface with real-time plotting
- **Programmatic API**: Full Python API for batch processing and automation
- **Example Scripts**: Comprehensive usage examples and best practices

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0
- scikit-learn >= 0.24.0
- tkinter (usually included with Python)
- pandas >= 1.1.0

## Quick Start

### GUI Application
```bash
python guinier_analysis.py
```

### Programmatic Usage

#### Basic Analysis
```python
from guinier_core import GuinierAnalyzer

# Initialize analyzer
analyzer = GuinierAnalyzer()

# Load data
result = analyzer.load_data('your_data.grad')

# Apply corrections
analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=3.0)

# Perform traditional fit
fit_result = analyzer.perform_fit(use_robust=True)

print(f"Rg = {fit_result['Rg']:.2f} ± {fit_result['Rg_error']:.2f} Å")
print(f"I0 = {fit_result['I0']:.2e} ± {fit_result['I0_error']:.2e}")
```

#### Scikit-Learn Enhanced Analysis
```python
from guinier_sklearn_integration import EnhancedGuinierAnalyzer

# Initialize enhanced analyzer
analyzer = EnhancedGuinierAnalyzer()

# Load data (same as above)
analyzer.load_data('your_data.grad')

# Compare multiple algorithms
comparison = analyzer.compare_methods()
analyzer.print_comparison()

# Use specific sklearn method
result = analyzer.fit_with_sklearn('huber', cross_validate=True)
print(f"Huber Regression: Rg = {result['Rg']:.2f} Å, CV Score = {np.mean(result['cv_scores']):.4f}")

# Get best model automatically
best_model = analyzer.get_best_sklearn_model()
print(f"Best model: {best_model['name']}")
```

## Architecture

### Modular Design
```
guinier_analysis.py          # Main entry point (GUI launcher)
├── guinier_core.py         # Core analysis engine
├── guinier_gui.py          # GUI interface
├── guinier_sklearn.py      # Scikit-learn integration
├── guinier_sklearn_integration.py  # Enhanced analyzer
└── example_usage.py        # Usage examples
```

### Core Components

#### 1. GuinierAnalyzer (guinier_core.py)
**Primary analysis engine with comprehensive SAXS processing capabilities**

**Key Methods:**
- `load_data(filename)`: Load SAXS data from various formats
- `apply_corrections(bg_value, norm_factor, snr_threshold)`: Data preprocessing
- `set_fit_range(q_min_idx, q_max_idx)`: Manual range selection
- `auto_fit_range()`: Automatic range detection using q·Rg ≤ 1.3
- `perform_fit(use_robust=True)`: Guinier fitting with robust options
- `export_results(filename)`: Save analysis results

#### 2. GuinierRegressor (guinier_sklearn.py)
**Scikit-learn compatible regressor for Guinier analysis**

**Features:**
- BaseEstimator and RegressorMixin inheritance
- Automatic Guinier parameter extraction (Rg, I0)
- Physical validation (q·Rg range checking)
- Compatible with sklearn pipelines and cross-validation

#### 3. SklearnGuinierAnalyzer (guinier_sklearn.py)
**Advanced analyzer using scikit-learn ecosystem**

**Capabilities:**
- Multiple regression algorithms comparison
- Hyperparameter optimization with GridSearchCV
- Cross-validation for model stability
- Automated model selection

#### 4. EnhancedGuinierAnalyzer (guinier_sklearn_integration.py)
**Unified interface combining traditional and ML approaches**

**Benefits:**
- Seamless integration with existing workflows
- Side-by-side method comparison
- Best practice recommendations
- Backward compatibility

## Algorithm Selection Guide

### Traditional Methods
| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **numpy.polyfit** | Clean data, standard analysis | Fast, well-understood | Sensitive to outliers |
| **Theil-Sen/Huber** | Data with outliers | Robust to outliers | Slower computation |

### Scikit-Learn Methods
| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **LinearRegression** | Clean data, baseline | Fast, identical to polyfit | No outlier robustness |
| **HuberRegressor** | Moderate outliers | Good balance of speed/robustness | Parameter tuning needed |
| **RANSACRegressor** | Many outliers | Excellent outlier rejection | Can be unstable |
| **TheilSenRegressor** | Robust analysis | Very robust, no parameters | Slow for large datasets |
| **Ridge** | Noisy data | Regularization prevents overfitting | May bias results |
| **Lasso** | Feature selection | Sparse solutions | Can be too aggressive |

### Recommendations
- **Default choice**: HuberRegressor (good balance of robustness and speed)
- **Clean data**: LinearRegression or numpy.polyfit
- **Many outliers**: RANSACRegressor or TheilSenRegressor
- **Noisy data**: Ridge regression
- **Uncertain**: Use `compare_methods()` to evaluate all algorithms

## GUI Usage

### Enhanced GUI Features
1. **Algorithm Selection**: Dropdown menu to choose regression method
2. **Method Comparison**: Button to compare all available algorithms
3. **Cross-Validation**: Enable/disable CV assessment
4. **Hyperparameter Tuning**: Automatic parameter optimization
5. **Real-time Results**: Live comparison of different methods

### GUI Workflow
1. **Load Data**: Use "Load SAXS Data" button
2. **Set Parameters**: Background, normalization, SNR threshold
3. **Choose Algorithm**: Select from dropdown (Linear, Huber, RANSAC, etc.)
4. **Set Range**: Manual or automatic (q·Rg ≤ 1.3)
5. **Perform Fit**: Click "Perform Guinier Fit"
6. **Compare Methods**: Use "Compare All Methods" for algorithm comparison
7. **Save Results**: Export results with all method comparisons

## API Reference

### Core Classes

#### GuinierAnalyzer
```python
class GuinierAnalyzer:
    def __init__(self)
    def load_data(self, filename: str) -> dict
    def apply_corrections(self, bg_value: float, norm_factor: float, snr_threshold: float) -> dict
    def set_fit_range(self, q_min_idx: int, q_max_idx: int) -> dict
    def auto_fit_range(self) -> dict
    def perform_fit(self, use_robust: bool = True) -> dict
    def export_results(self, filename: str) -> dict
```

#### GuinierRegressor
```python
class GuinierRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor=None, validate_range=True, max_qrg=1.3)
    def fit(self, X, y, sample_weight=None)
    def predict(self, X)
    def score(self, X, y, sample_weight=None)
    
    # Fitted parameters
    .Rg_: float              # Radius of gyration
    .I0_: float              # Forward scattering intensity
    .max_qrg_actual_: float  # Maximum q·Rg value
```

#### SklearnGuinierAnalyzer
```python
class SklearnGuinierAnalyzer:
    def __init__(self)
    def load_data(self, q, I, dI=None)
    def fit_multiple_models(self, q_range=None, use_weights=True, cv_folds=5) -> dict
    def hyperparameter_tuning(self, regressor_name='huber', param_grid=None, cv_folds=5) -> dict
    def plot_comparison(self, save_filename=None)
    def get_summary_table(self) -> pd.DataFrame
```

#### EnhancedGuinierAnalyzer
```python
class EnhancedGuinierAnalyzer(GuinierAnalyzer):
    def fit_with_sklearn(self, regressor_name='huber', cross_validate=True) -> dict
    def compare_methods(self) -> dict
    def get_best_sklearn_model(self) -> dict
    def print_comparison(self)
```

## Example Workflows

### 1. Single File Analysis
```python
from guinier_core import GuinierAnalyzer

analyzer = GuinierAnalyzer()
result = analyzer.load_data('sample.grad')
analyzer.apply_corrections(bg_value=0.1, norm_factor=1.0, snr_threshold=3.0)
analyzer.auto_fit_range()
fit_result = analyzer.perform_fit(use_robust=True)

print(f"Rg = {fit_result['Rg']:.2f} ± {fit_result['Rg_error']:.2f} Å")
```

### 2. Batch Processing with Algorithm Comparison
```python
from guinier_sklearn_integration import EnhancedGuinierAnalyzer
import glob

results = []
for filename in glob.glob('*.grad'):
    analyzer = EnhancedGuinierAnalyzer()
    analyzer.load_data(filename)
    analyzer.apply_corrections(0.1, 1.0, 3.0)
    analyzer.auto_fit_range()
    
    # Compare methods
    comparison = analyzer.compare_methods()
    best_model = analyzer.get_best_sklearn_model()
    
    results.append({
        'filename': filename,
        'best_method': best_model['name'],
        'Rg': best_model['result']['Rg'],
        'cv_score': best_model['cv_score']
    })

import pandas as pd
df = pd.DataFrame(results)
df.to_csv('batch_results.csv', index=False)
```

### 3. Hyperparameter Optimization
```python
from guinier_sklearn import SklearnGuinierAnalyzer

analyzer = SklearnGuinierAnalyzer()
analyzer.load_data(q, I, dI)

# Tune Huber regression
param_grid = {
    'guinier__regressor__epsilon': [1.1, 1.35, 1.5, 2.0],
    'guinier__regressor__alpha': [1e-4, 1e-3, 1e-2]
}

tuning_results = analyzer.hyperparameter_tuning('huber', param_grid, cv_folds=5)
print(f"Best parameters: {tuning_results['best_params']}")
print(f"Best Rg: {tuning_results['Rg']:.2f} Å")
```

## Physical Validation

### Guinier Regime Validity
The Guinier approximation is valid when **q·Rg ≤ 1.3**. The software automatically:
- Checks this condition for all fits
- Warns when the limit is exceeded
- Provides automatic range selection to ensure validity
- Marks invalid fits in comparison tables

### Quality Metrics
- **R²**: Goodness of fit for ln(I) vs q² (should be > 0.95)
- **χ²**: Reduced chi-squared (should be close to 1.0)
- **Residual Analysis**: Systematic deviations from linear behavior
- **Cross-Validation**: Model stability and generalization

## Troubleshooting

### Common Issues
1. **Low R² values**: Check data quality, background subtraction, q-range
2. **q·Rg > 1.3**: Reduce q-range or check if data is suitable for Guinier analysis
3. **Unstable fits**: Try robust methods (Huber, Theil-Sen) or increase SNR threshold
4. **Algorithm comparison shows large differences**: Indicates data quality issues

### Best Practices
1. **Always check q·Rg ≤ 1.3** for physical validity
2. **Use robust methods** for real experimental data
3. **Compare multiple algorithms** to assess result stability
4. **Validate with cross-validation** for important measurements
5. **Export full results** including all quality metrics

## Contributing

### Development Setup
```bash
git clone https://github.com/your-repo/guinier-analysis.git
cd guinier-analysis
pip install -r requirements.txt
```

### Testing
```bash
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Document all public methods
- Include comprehensive docstrings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{guinier_analysis,
  title={Guinier Analysis for SAXS Data with Machine Learning Integration},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/guinier-analysis}
}
```

## References

1. Guinier, A. & Fournet, G. (1955). Small-Angle Scattering of X-rays. Wiley.
2. Hammersley, A. P. (2016). FIT2D: a multi-purpose data reduction, analysis and visualization program. J. Appl. Cryst. 49, 646-652.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825-2830.

---

For more information, examples, and updates, visit our [GitHub repository](https://github.com/your-repo/guinier-analysis). 
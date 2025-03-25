# Guinier Analysis for SAXS Data

A graphical user interface (GUI) application for analyzing Small Angle X-ray Scattering (SAXS) data using the Guinier approximation.

## Overview

This application allows researchers to:
- Load and visualize SAXS data
- Perform background subtraction and normalization
- Apply the Guinier approximation to extract radius of gyration (Rg) values
- Automatically determine the valid Guinier regime (q·Rg ≤ 1.3)
- Visualize results through interactive plots

## Requirements

- Python 3.6+
- Dependencies: numpy, matplotlib, scipy, pandas, tkinter

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
python guinier_analysis.py
```

2. Load your SAXS data (two-column format: q, I)
3. Set background and normalization factors if needed
4. Use "Auto Range" to automatically determine the valid Guinier regime (q·Rg ≤ 1.3)
5. Click "Perform Guinier Fit" to calculate Rg and I₀

## Sample Data

For testing purposes, you can generate synthetic SAXS data:

```bash
python generate_sample_data.py
```

This will create sample data files in the `sample_data` directory with different Rg values (4.2Å, 8.0Å, and 12.5Å) to simulate time-dependent evolution.

## Theoretical Background

The Guinier approximation describes the scattering intensity I(q) in the low-q region:

I(q) = I₀ · exp(-q²Rg²/3)

Where:
- I₀ is the extrapolated intensity at q=0
- Rg is the radius of gyration of the scattering domains
- The approximation is valid in the region where q·Rg ≤ 1.3

By plotting ln(I) vs q², a linear relationship is expected in the Guinier regime, with:
- Slope = -Rg²/3
- Intercept = ln(I₀)

## Notes

- Data should be properly background-subtracted and normalized before analysis
- The Guinier approximation is only valid for dilute, non-interacting systems
- The validity condition (q·Rg ≤ 1.3) should be verified for reliable results 
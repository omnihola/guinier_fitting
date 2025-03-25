import numpy as np
import pandas as pd
import os

def generate_saxs_data(output_file, Rg=12.5, I0=1000.0, q_min=0.01, q_max=0.3, noise_level=0.05, num_points=100, bg_level=5.0):
    """
    Generate synthetic SAXS data following the Guinier approximation
    
    Parameters:
    ----------
    output_file : str
        Path to save the generated data
    Rg : float
        Radius of gyration in Angstroms
    I0 : float
        Zero-angle scattering intensity
    q_min, q_max : float
        Minimum and maximum q values
    noise_level : float
        Relative noise level (0.1 = 10% noise)
    num_points : int
        Number of data points to generate
    bg_level : float
        Background level to add
    """
    # Generate q values (log spacing is typical for SAXS data)
    q = np.logspace(np.log10(q_min), np.log10(q_max), num_points)
    
    # Calculate intensity using Guinier approximation
    I = I0 * np.exp(-(q**2) * (Rg**2) / 3.0) + bg_level
    
    # Add random noise
    noise = np.random.normal(0, noise_level * I, num_points)
    I_noisy = I + noise
    
    # Ensure all intensities are positive
    I_noisy = np.maximum(I_noisy, 1e-10)
    
    # Save to file
    data = np.column_stack((q, I_noisy))
    pd.DataFrame(data).to_csv(output_file, header=False, index=False)
    
    print(f"Generated SAXS data with Rg = {Rg} Å, saved to {output_file}")
    return q, I_noisy

if __name__ == "__main__":
    # Generate sample data for different Rg values to simulate time-dependent evolution
    os.makedirs("sample_data", exist_ok=True)
    
    # Sample for x=1h (Rg = 4.2 nm)
    generate_saxs_data("sample_data/saxs_data_1h.csv", Rg=4.2, I0=800.0, bg_level=2.0)
    
    # Sample for x=3h (intermediate)
    generate_saxs_data("sample_data/saxs_data_3h.csv", Rg=8.0, I0=900.0, bg_level=3.0)
    
    # Sample for x=6h (Rg = 12.5 nm)
    generate_saxs_data("sample_data/saxs_data_6h.csv", Rg=12.5, I0=1000.0, bg_level=5.0)
    
    print("All sample data generated successfully.") 
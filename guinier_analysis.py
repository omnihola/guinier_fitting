#!/usr/bin/env python3
"""
Guinier Analysis for SAXS Data - Main Application Entry Point

This is the main entry point for the Guinier Analysis GUI application.
The application has been refactored into modular components:
- guinier_core.py: Core analysis functionality
- guinier_gui.py: GUI components

For backward compatibility, this script can still be run directly.
"""

import sys
import os

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from guinier_gui import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1) 
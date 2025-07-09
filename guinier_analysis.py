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
    import tkinter as tk
    from guinier_gui import GuinierAnalysisGUI
    
    def main():
        """Main function to launch the Guinier analysis GUI with icon support."""
        root = tk.Tk()
        
        # Set application icon at startup
        try:
            if os.path.exists('guinier_icon.png'):
                # For most systems, use PNG
                icon = tk.PhotoImage(file='guinier_icon.png')
                root.iconphoto(True, icon)
            elif os.path.exists('guinier_icon.ico'):
                # For Windows systems
                root.iconbitmap('guinier_icon.ico')
        except Exception as e:
            print(f"Could not load icon: {e}")
        
        app = GuinierAnalysisGUI(root)
        root.mainloop()
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1) 
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

def main():
    """Main function to launch the Guinier analysis GUI."""
    print("Starting Guinier Analysis GUI...")
    
    # Try modern PySide6 interface first
    try:
        from guinier_gui_pyside6 import main as pyside6_main
        print("Using modern PySide6 interface")
        pyside6_main()
        return
    except ImportError:
        print("PySide6 not available, falling back to tkinter interface")
    
    # Fallback to tkinter interface
    try:
        import tkinter as tk
        from guinier_gui import GuinierAnalysisGUI
        
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
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Neither PySide6 nor tkinter are available.")
        print("Please install PySide6 for the best experience:")
        print("pip install PySide6")
        print("Or ensure tkinter is available on your system.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
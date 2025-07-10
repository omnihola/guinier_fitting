import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import os

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QWidget, QLabel, QLineEdit, QPushButton, QComboBox, 
                               QCheckBox, QTextEdit, QScrollArea, QGroupBox, QFormLayout,
                               QFileDialog, QMessageBox, QSplitter, QTabWidget, QProgressBar,
                               QStatusBar, QFrame, QGridLayout, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QFont, QPixmap, QPalette, QColor

from guinier_core import GuinierAnalyzer


class ModernGuinierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guinier Analysis for SAXS Data - Enhanced with Machine Learning")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize analyzer
        self.analyzer = GuinierAnalyzer()
        
        # Set application style
        self.setStyleSheet(self.get_modern_style())
        
        # Set application icon
        self.set_app_icon()
        
        # Initialize UI
        self.init_ui()
        
        # Initialize plots
        self.init_plots()
        
        # Setup status bar
        self.setup_status_bar()
        
    def set_app_icon(self):
        """Set application icon"""
        try:
            if os.path.exists('guinier_icon.png'):
                self.setWindowIcon(QIcon('guinier_icon.png'))
            elif os.path.exists('guinier_icon.ico'):
                self.setWindowIcon(QIcon('guinier_icon.ico'))
        except Exception as e:
            print(f"Could not load icon: {e}")
    
    def get_modern_style(self):
        """Return modern CSS styling"""
        return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: white;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #2c3e50;
        }
        
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        
        QPushButton:hover {
            background-color: #2980b9;
        }
        
        QPushButton:pressed {
            background-color: #21618c;
        }
        
        QPushButton:disabled {
            background-color: #bdc3c7;
        }
        
        QLineEdit {
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
        }
        
        QLineEdit:focus {
            border: 2px solid #3498db;
        }
        
        QComboBox {
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
            min-width: 100px;
        }
        
        QComboBox::drop-down {
            border: none;
            background-color: #3498db;
            border-radius: 4px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border: none;
            background-color: #3498db;
        }
        
        QTextEdit {
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10px;
        }
        
        QCheckBox {
            spacing: 5px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 3px;
            border: 1px solid #ddd;
            background-color: white;
        }
        
        QCheckBox::indicator:checked {
            background-color: #3498db;
            border: 1px solid #3498db;
        }
        
        QLabel {
            color: #2c3e50;
        }
        
        QStatusBar {
            background-color: #34495e;
            color: white;
            border: none;
        }
        
        QProgressBar {
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #ecf0f1;
            text-align: center;
        }
        
        QProgressBar::chunk {
            background-color: #3498db;
            border-radius: 4px;
        }
        
        QSplitter::handle {
            background-color: #bdc3c7;
            width: 3px;
        }
        
        QSplitter::handle:hover {
            background-color: #95a5a6;
        }
        """
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (controls)
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Right panel (plots)
        plot_panel = self.create_plot_panel()
        splitter.addWidget(plot_panel)
        
        # Set initial sizes (30% controls, 70% plots)
        splitter.setSizes([400, 1200])
        
    def create_control_panel(self):
        """Create the control panel"""
        control_widget = QWidget()
        control_widget.setMaximumWidth(450)
        control_widget.setMinimumWidth(400)
        
        layout = QVBoxLayout(control_widget)
        
        # Data Loading Group
        self.create_data_loading_group(layout)
        
        # Data Processing Group
        self.create_data_processing_group(layout)
        
        # Fitting Range Group
        self.create_fitting_range_group(layout)
        
        # Fitting Group
        self.create_fitting_group(layout)
        
        # Results Group
        self.create_results_group(layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return control_widget
    
    def create_data_loading_group(self, parent_layout):
        """Create data loading group"""
        group = QGroupBox("Data Loading")
        layout = QVBoxLayout(group)
        
        # Load button
        self.load_btn = QPushButton("Load SAXS Data")
        self.load_btn.clicked.connect(self.load_data)
        layout.addWidget(self.load_btn)
        
        # Format label
        format_label = QLabel("Supported formats: .grad, .txt, .csv (q, I columns)")
        format_label.setWordWrap(True)
        format_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        layout.addWidget(format_label)
        
        parent_layout.addWidget(group)
    
    def create_data_processing_group(self, parent_layout):
        """Create data processing group"""
        group = QGroupBox("Data Processing")
        layout = QFormLayout(group)
        
        # Background subtraction
        self.bg_edit = QLineEdit("0.0")
        layout.addRow("Background:", self.bg_edit)
        
        # Normalization
        self.norm_edit = QLineEdit("1.0")
        layout.addRow("Norm Factor:", self.norm_edit)
        
        # SNR Filtering
        self.snr_edit = QLineEdit("3.0")
        layout.addRow("Min SNR (I/σ):", self.snr_edit)
        
        # Apply button
        self.apply_corrections_btn = QPushButton("Apply Corrections")
        self.apply_corrections_btn.clicked.connect(self.apply_corrections)
        layout.addRow(self.apply_corrections_btn)
        
        parent_layout.addWidget(group)
    
    def create_fitting_range_group(self, parent_layout):
        """Create fitting range group"""
        group = QGroupBox("Fitting Range")
        layout = QFormLayout(group)
        
        # Range inputs
        self.q_min_edit = QLineEdit("0")
        layout.addRow("q_min index:", self.q_min_edit)
        
        self.q_max_edit = QLineEdit("-1")
        layout.addRow("q_max index:", self.q_max_edit)
        
        # Range buttons
        button_layout = QHBoxLayout()
        
        self.set_range_btn = QPushButton("Set Range")
        self.set_range_btn.clicked.connect(self.set_fit_range)
        button_layout.addWidget(self.set_range_btn)
        
        self.auto_range_btn = QPushButton("Auto Range")
        self.auto_range_btn.clicked.connect(self.auto_range)
        button_layout.addWidget(self.auto_range_btn)
        
        layout.addRow(button_layout)
        
        parent_layout.addWidget(group)
    
    def create_fitting_group(self, parent_layout):
        """Create fitting group"""
        group = QGroupBox("Guinier Fitting")
        layout = QFormLayout(group)
        
        # Algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "huber", "traditional", "traditional_robust", 
            "linear", "ridge", "theilsen"
        ])
        layout.addRow("Algorithm:", self.algorithm_combo)
        
        # Cross-validation option
        self.cv_checkbox = QCheckBox("Cross-validation")
        self.cv_checkbox.setChecked(True)
        layout.addRow(self.cv_checkbox)
        
        # Robust fitting option
        self.robust_checkbox = QCheckBox("Robust fitting (traditional)")
        self.robust_checkbox.setChecked(True)
        layout.addRow(self.robust_checkbox)
        
        # Fitting buttons
        button_layout = QVBoxLayout()
        
        self.fit_btn = QPushButton("Perform Guinier Fit")
        self.fit_btn.clicked.connect(self.perform_fit)
        button_layout.addWidget(self.fit_btn)
        
        self.compare_btn = QPushButton("Compare All Methods")
        self.compare_btn.clicked.connect(self.compare_methods)
        button_layout.addWidget(self.compare_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        button_layout.addWidget(self.save_btn)
        
        layout.addRow(button_layout)
        
        parent_layout.addWidget(group)
    
    def create_results_group(self, parent_layout):
        """Create results group"""
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(300)
        layout.addWidget(self.results_text)
        
        parent_layout.addWidget(group)
    
    def create_plot_panel(self):
        """Create the plot panel"""
        plot_widget = QWidget()
        layout = QVBoxLayout(plot_widget)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        layout.addWidget(self.toolbar)
        
        return plot_widget
    
    def init_plots(self):
        """Initialize the plots"""
        # Create subplots
        self.ax1 = self.figure.add_subplot(311)
        self.ax1.set_title("SAXS Data")
        self.ax1.set_xlabel("q (Å⁻¹)")
        self.ax1.set_ylabel("I(q)")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2 = self.figure.add_subplot(312)
        self.ax2.set_title("Guinier Plot: ln(I) vs q²")
        self.ax2.set_xlabel("q² (Å⁻²)")
        self.ax2.set_ylabel("ln(I)")
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3 = self.figure.add_subplot(313)
        self.ax3.set_title("Residuals")
        self.ax3.set_xlabel("q² (Å⁻²)")
        self.ax3.set_ylabel("Residuals")
        self.ax3.grid(True, alpha=0.3)
        self.ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.status_bar.showMessage("Ready")
    
    def show_progress(self, message="Processing..."):
        """Show progress indication"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_bar.showMessage(message)
    
    def hide_progress(self):
        """Hide progress indication"""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Ready")
    
    def load_data(self):
        """Load SAXS data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SAXS Data File", "",
            "GRAD Files (*.grad);;Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        self.show_progress("Loading data...")
        
        try:
            # Use core analyzer to load data
            result = self.analyzer.load_data(file_path)
            
            if result['success']:
                # Update GUI fields
                self.q_min_edit.setText(str(self.analyzer.q_min_idx))
                self.q_max_edit.setText(str(self.analyzer.q_max_idx))
                self.bg_edit.setText(str(self.analyzer.bg_value))
                self.norm_edit.setText(str(self.analyzer.norm_factor))
                self.snr_edit.setText(str(self.analyzer.snr_threshold))
                
                # Update plots
                self.update_plots()
                
                QMessageBox.information(self, "Success", result['message'])
                self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            else:
                QMessageBox.critical(self, "Error", result['message'])
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
        finally:
            self.hide_progress()
    
    def apply_corrections(self):
        """Apply data corrections"""
        try:
            bg_value = float(self.bg_edit.text())
            norm_factor = float(self.norm_edit.text())
            snr_threshold = float(self.snr_edit.text())
            
            self.show_progress("Applying corrections...")
            
            # Use core analyzer to apply corrections
            result = self.analyzer.apply_corrections(bg_value, norm_factor, snr_threshold)
            
            if result['success']:
                # Update GUI fields
                self.q_min_edit.setText(str(self.analyzer.q_min_idx))
                self.q_max_edit.setText(str(self.analyzer.q_max_idx))
                
                self.update_plots()
                QMessageBox.information(self, "Success", result['message'])
            else:
                QMessageBox.critical(self, "Error", result['message'])
                
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid correction parameters")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply corrections: {str(e)}")
        finally:
            self.hide_progress()
    
    def set_fit_range(self):
        """Set fitting range"""
        try:
            q_min_idx = int(self.q_min_edit.text())
            q_max_idx = int(self.q_max_edit.text())
            
            self.show_progress("Setting fit range...")
            
            # Use core analyzer to set range
            result = self.analyzer.set_fit_range(q_min_idx, q_max_idx)
            
            if result['success']:
                self.update_plots()
                QMessageBox.information(self, "Success", result['message'])
            else:
                QMessageBox.critical(self, "Error", result['message'])
                
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid fit range parameters")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set fit range: {str(e)}")
        finally:
            self.hide_progress()
    
    def auto_range(self):
        """Auto-set fitting range"""
        try:
            self.show_progress("Auto-setting range...")
            
            # Use core analyzer to auto-set range
            result = self.analyzer.auto_range()
            
            if result['success']:
                # Update GUI fields
                self.q_min_edit.setText(str(result['q_min_idx']))
                self.q_max_edit.setText(str(result['q_max_idx']))
                
                self.update_plots()
                message = f"{result['message']} (Estimated Rg: {result['initial_Rg']:.2f} Å)"
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", result['message'])
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to auto-set range: {str(e)}")
        finally:
            self.hide_progress()
    
    def perform_fit(self):
        """Perform Guinier fitting"""
        try:
            algorithm = self.algorithm_combo.currentText()
            
            self.show_progress("Performing fit...")
            
            if algorithm == "traditional":
                # Traditional numpy polyfit
                result = self.analyzer.perform_fit(use_robust=False)
            elif algorithm == "traditional_robust":
                # Traditional robust fit
                result = self.analyzer.perform_fit(use_robust=True)
            else:
                # Scikit-learn methods
                result = self.analyzer.fit_with_sklearn(algorithm, cross_validate=self.cv_checkbox.isChecked())
            
            if result['success']:
                # Update plots with fit
                self.update_plots(show_fit=True)
                
                # Update results display
                self.update_results()
                
                message = result['message']
                if result.get('warning'):
                    message += f"\n\n{result['warning']}"
                
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", result['message'])
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Fitting failed: {str(e)}")
        finally:
            self.hide_progress()
    
    def compare_methods(self):
        """Compare all available fitting methods"""
        try:
            self.show_progress("Comparing methods...")
            
            # Perform comparison
            comparison_result = self.analyzer.compare_methods()
            
            if comparison_result['success']:
                # Update plots with comparison
                self.update_plots(show_fit=True)
                
                # Show comparison results in a new window
                self.show_comparison_window()
                
                QMessageBox.information(self, "Success", "Method comparison completed!")
            else:
                QMessageBox.critical(self, "Error", comparison_result['message'])
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed: {str(e)}")
        finally:
            self.hide_progress()
    
    def show_comparison_window(self):
        """Show method comparison results in a new window"""
        if self.analyzer.model_comparison is None:
            QMessageBox.warning(self, "Warning", "No comparison results available.")
            return
        
        # Create comparison dialog
        dialog = ComparisonDialog(self.analyzer, self)
        dialog.exec()
        
        # Update algorithm selection to best model if available
        best_model = self.analyzer.get_best_sklearn_model()
        if best_model:
            # Find and select the best model in combo box
            index = self.algorithm_combo.findText(best_model['name'])
            if index >= 0:
                self.algorithm_combo.setCurrentIndex(index)
    
    def save_results(self):
        """Save results"""
        fit_results = self.analyzer.get_fit_results()
        if not fit_results:
            QMessageBox.warning(self, "Warning", "No fitting results to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Fitting Results", "",
            "CSV Files (*.csv);;Text Files (*.txt)"
        )
        
        if not file_path:
            return
        
        try:
            self.show_progress("Saving results...")
            
            # Use core analyzer to save results
            result = self.analyzer.save_results(file_path)
            
            if result['success']:
                # Try to save plot
                try:
                    report_filename = os.path.splitext(file_path)[0] + "_report.pdf"
                    self.figure.savefig(report_filename, dpi=300, bbox_inches='tight')
                    message = f"{result['message']}\nData saved to {result.get('data_file', 'N/A')}\nPlots saved to {report_filename}"
                except Exception as e:
                    message = f"{result['message']}\nData saved to {result.get('data_file', 'N/A')}\nCould not save plots: {str(e)}"
                
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", result['message'])
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
        finally:
            self.hide_progress()
    
    def update_plots(self, show_fit=False):
        """Update all plots"""
        # Get processed data from analyzer
        data = self.analyzer.get_processed_data()
        if not data:
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Set titles and labels
        self.ax1.set_title("SAXS Data", fontsize=12, fontweight='bold')
        self.ax1.set_xlabel("q (Å⁻¹)")
        self.ax1.set_ylabel("I(q)")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("Guinier Plot: ln(I) vs q²", fontsize=12, fontweight='bold')
        self.ax2.set_xlabel("q² (Å⁻²)")
        self.ax2.set_ylabel("ln(I)")
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title("Residuals", fontsize=12, fontweight='bold')
        self.ax3.set_xlabel("q² (Å⁻²)")
        self.ax3.set_ylabel("ln(I) - fit")
        self.ax3.grid(True, alpha=0.3)
        self.ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot data
        self._plot_saxs_data(data)
        self._plot_guinier_data(data)
        
        # Show fit if available
        if show_fit:
            self._plot_fit_results(data)
        
        # Add legends
        self.ax1.legend(loc='best', fontsize=9)
        self.ax2.legend(loc='best', fontsize=9)
        if show_fit:
            self.ax3.legend(loc='best', fontsize=9)
        
        # Adjust layout
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _plot_saxs_data(self, data):
        """Plot SAXS data on the first subplot"""
        # Plot original data with error bars if available
        if data.get('dI_original') is not None:
            # All data with error bars
            dI_corrected = data['dI_original'] / data['norm_factor']
            self.ax1.errorbar(data['q_original'], data['I_corrected_all'], yerr=dI_corrected, 
                             fmt='o', markersize=2, elinewidth=0.5, capsize=1, alpha=0.3,
                             label='All Data', color='lightblue')
        else:
            # No error data available
            self.ax1.plot(data['q_original'], data['I_corrected_all'], 'o', markersize=2, 
                         alpha=0.3, label='All Data', color='lightblue')
        
        # Plot filtered data if available
        if 'q_filtered' in data:
            if data.get('dI_filtered') is not None:
                dI_filtered = data['dI_filtered'] / data['norm_factor']
                self.ax1.errorbar(data['q_filtered'], data['I_corrected_filtered'], yerr=dI_filtered, 
                                 fmt='o', markersize=4, elinewidth=1, capsize=2, color='blue',
                                 label=f'Filtered Data (SNR ≥ {data["snr_threshold"]})')
            else:
                self.ax1.plot(data['q_filtered'], data['I_corrected_filtered'], 'o', markersize=4, 
                             label='Filtered Data', color='blue')
        
        # Highlight fit range
        if 'q_range' in data:
            if data.get('dI_range') is not None:
                dI_range = data['dI_range'] / data['norm_factor']
                self.ax1.errorbar(data['q_range'], data['I_corrected_range'], yerr=dI_range, 
                                 fmt='o', markersize=6, color='red', elinewidth=1, capsize=2,
                                 label='Fit Range')
            else:
                self.ax1.plot(data['q_range'], data['I_corrected_range'], 'o', markersize=6, 
                             color='red', label='Fit Range')
        
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
                         'o', markersize=2, alpha=0.3, label='All Data', color='lightblue')
        
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
                                     label='Filtered Data', color='blue')
                else:
                    self.ax2.plot(q_sq_filtered[valid_idx_filtered], ln_I_filtered[valid_idx_filtered], 
                                 'o', markersize=4, label='Filtered Data', color='blue')
        
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
        self.ax2.set_title(f"Guinier Plot: ln(I) vs q² [ln(I₀)={fit_results['fit_intercept']:.2f}, Rg={fit_results['Rg']:.2f} Å]", 
                          fontsize=12, fontweight='bold')
        
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
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
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
        
        # Clear previous results
        self.results_text.clear()
        
        # Add results
        algorithm = self.algorithm_combo.currentText()
        results_html = f"""
        <h3>Guinier Analysis Results ({algorithm})</h3>
        <hr>
        <table width="100%">
        <tr><td><b>Radius of Gyration (Rg):</b></td><td>{fit_results['Rg']:.2f} ± {fit_results['Rg_error']:.2f} Å</td></tr>
        <tr><td><b>Zero-angle Intensity (I₀):</b></td><td>{fit_results['I0']:.2e} ± {fit_results['I0_error']:.2e}</td></tr>
        """
        
        # Add validation metrics
        if fit_results['r_squared'] is not None:
            r2_status = "Good" if fit_results['r_squared'] > 0.99 else "Poor"
            r2_color = "green" if fit_results['r_squared'] > 0.99 else "red"
            results_html += f'<tr><td><b>R² (Goodness of fit):</b></td><td><span style="color:{r2_color}">{fit_results["r_squared"]:.4f} ({r2_status})</span></td></tr>'
        
        if fit_results['chi_squared'] is not None:
            chi2_status = "Good" if 0.5 <= fit_results['chi_squared'] <= 1.5 else "Poor"
            chi2_color = "green" if 0.5 <= fit_results['chi_squared'] <= 1.5 else "red"
            results_html += f'<tr><td><b>χ²ᵣₑₙ (Reduced chi-squared):</b></td><td><span style="color:{chi2_color}">{fit_results["chi_squared"]:.4f} ({chi2_status})</span></td></tr>'
        
        # Add sklearn-specific results
        if hasattr(self.analyzer, 'sklearn_models') and algorithm in self.analyzer.sklearn_models:
            sklearn_result = self.analyzer.sklearn_models[algorithm]
            if sklearn_result.get('cv_scores') is not None:
                cv_scores = sklearn_result['cv_scores']
                cv_status = "Good" if np.mean(cv_scores) > 0.9 else "Poor"
                cv_color = "green" if np.mean(cv_scores) > 0.9 else "red"
                results_html += f'<tr><td><b>Cross-validation Score:</b></td><td>{np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}</td></tr>'
                results_html += f'<tr><td><b>Model Stability:</b></td><td><span style="color:{cv_color}">{cv_status}</span></td></tr>'
        
        # Add fitting range info
        if fit_results['max_q_rg'] is not None:
            q_rg_status = "Valid" if fit_results['max_q_rg'] <= 1.3 else "Exceeds limit"
            q_rg_color = "green" if fit_results['max_q_rg'] <= 1.3 else "red"
            results_html += f'<tr><td><b>Fitting Range:</b></td><td><span style="color:{q_rg_color}">q·Rg ≤ {fit_results["max_q_rg"]:.2f} ({q_rg_status})</span></td></tr>'
        
        data = self.analyzer.get_processed_data()
        if data and 'q_range' in data:
            results_html += f'<tr><td><b>q range:</b></td><td>{data["q_range"][0]:.4f} - {data["q_range"][-1]:.4f} Å⁻¹</td></tr>'
        
        results_html += '</table>'
        
        # Add algorithm notes
        results_html += '<h4>Algorithm Notes:</h4>'
        algorithm_notes = {
            'traditional': "Standard numpy.polyfit - fast but sensitive to outliers",
            'traditional_robust': "Robust methods (Theil-Sen/Huber) - good for outliers",
            'linear': "sklearn LinearRegression - equivalent to numpy.polyfit",
            'huber': "Huber regression - robust to moderate outliers (recommended)",
            'ridge': "Ridge regression - good for noisy data with regularization",
            'theilsen': "Theil-Sen regression - very robust to outliers but slower"
        }
        note = algorithm_notes.get(algorithm, "Unknown algorithm")
        results_html += f'<p>• {note}</p>'
        
        # Add physical validation
        if fit_results['Rg'] > 0:
            results_html += '''
            <h4>Physical Validation:</h4>
            <ul>
            <li>Globular proteins: Rg ≈ 0.77 × (MW in kDa)^(1/3) nm</li>
            <li>Extended proteins: Rg may be 1.5-2× larger</li>
            <li>Verify results with literature or known structures</li>
            </ul>
            '''
        
        # Best model recommendation
        if hasattr(self.analyzer, 'model_comparison') and self.analyzer.model_comparison:
            best_model = self.analyzer.get_best_sklearn_model()
            if best_model and best_model['name'] != algorithm:
                results_html += f'<p><b>Recommendation:</b> Try "{best_model["name"]}" method (CV score: {best_model["cv_score"]:.4f})</p>'
        
        self.results_text.setHtml(results_html)


class ComparisonDialog(QMainWindow):
    """Dialog for displaying method comparison results"""
    
    def __init__(self, analyzer, parent=None):
        super().__init__(parent)
        self.analyzer = analyzer
        self.setWindowTitle("Method Comparison Results")
        self.setGeometry(200, 200, 1000, 800)
        
        # Set up the UI
        self.setup_ui()
        
        # Display results
        self.display_comparison_results()
    
    def setup_ui(self):
        """Set up the comparison dialog UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("Method Comparison Results")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; font-size: 11px;")
        layout.addWidget(self.results_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def display_comparison_results(self):
        """Display the comparison results"""
        if self.analyzer.model_comparison is None:
            self.results_text.setPlainText("No comparison results available.")
            return
        
        # Format and display comparison results
        comparison_text = self.format_comparison_results()
        self.results_text.setPlainText(comparison_text)
    
    def format_comparison_results(self):
        """Format comparison results for display"""
        text = "METHOD COMPARISON RESULTS\n"
        text += "=" * 80 + "\n\n"
        
        # Header
        text += f"{'Method':<20} {'Rg (Å)':<10} {'I0':<12} {'R²':<8} {'χ²':<8} {'CV_mean':<8} {'CV_std':<8} {'Valid':<6}\n"
        text += "-" * 80 + "\n"
        
        # Results
        for name, result in self.analyzer.model_comparison.items():
            cv_mean = f"{result.get('cv_mean', 0):.4f}" if result.get('cv_mean') is not None else "N/A"
            cv_std = f"{result.get('cv_std', 0):.4f}" if result.get('cv_std') is not None else "N/A"
            valid = "Yes" if result.get('valid_guinier', False) else "No"
            
            text += f"{result['method']:<20} "
            text += f"{result['Rg']:<10.3f} "
            text += f"{result['I0']:<12.2e} "
            text += f"{result['r_squared']:<8.4f} "
            text += f"{result['chi_squared']:<8.4f} "
            text += f"{cv_mean:<8} "
            text += f"{cv_std:<8} "
            text += f"{valid:<6}\n"
        
        text += "\n" + "=" * 80 + "\n"
        
        # Recommendations
        text += "RECOMMENDATIONS:\n"
        text += "• Use 'huber' for data with moderate outliers (recommended default)\n"
        text += "• Use 'linear' for clean data with minimal outliers\n"
        text += "• Use 'theilsen' for data with many outliers\n"
        text += "• Use 'ridge' for noisy data (regularization helps)\n"
        text += "• Traditional methods provide good baseline comparison\n"
        text += "• Cross-validation scores indicate model stability\n"
        text += "• Valid = Yes means q·Rg ≤ 1.3 (physically meaningful)\n"
        
        return text


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Guinier Analysis")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Scientific Computing")
    
    # Create and show the main window
    window = ModernGuinierGUI()
    window.show()
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
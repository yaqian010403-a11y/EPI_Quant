import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import cv2
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel, \
    QMessageBox, QCheckBox, QDialog, QLineEdit, QTabWidget, QSizePolicy, QGroupBox, QGridLayout, QTextEdit, QProgressDialog
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import config
from gui_cell_track import ComparisonMainWindow  # Import the tracking results display and correction window
from gui_cell_direction import CellDirection  # Import the tracking results display and correction window
from gui_plot_track import MainWindow as PlotTrackWindow  # Import the trajectory display window
from gui_comparison import MainWindow as CellClassification  # Import the cell correction window
from gui_quantitative_analysis import QuantitativeAnalysisGUI
from gui_cell_shadow_plot import CellShadowLineGUI
import shutil
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QScrollBar
import matplotlib.backends.backend_pdf


class ReportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Classification Report")
        self.setGeometry(100, 100, 1000, 800)

        # Create layout
        layout = QVBoxLayout(self)

        # Create a layout for displaying images (two rows and two columns)
        grid_layout = QGridLayout()

        # Get image paths
        image_folder = config.cell_classification_output_path
        #print(image_folder)
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Only take the first four images
        images_to_display = image_files[:]

        # Iterate through images, create labels, and add them to the grid layout
        for i, image_file in enumerate(images_to_display):
            image_path = os.path.join(image_folder, image_file)
            img_data = cv2.imread(image_path)
            if img_data is not None:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                canvas = MplCanvas(self, width=4, height=3, dpi=100)
                ax = canvas.fig.add_subplot(111)
                ax.imshow(img_data)
                ax.axis('off')
                row, col = divmod(i, 2)
                grid_layout.addWidget(canvas, row, col)

        layout.addLayout(grid_layout)



class FeatureSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Selection")
        self.setGeometry(100, 100, 1000, 500)  # Adjust the width to accommodate images

        # Set the main layout to horizontal layout
        main_layout = QHBoxLayout()

        # Left: Feature selection part
        feature_layout = QVBoxLayout()

        # Add a checkbox for excluding edge cells
        self.exclude_edge_cells_checkbox = QCheckBox("Exclude Edge Cells")
        feature_layout.addWidget(self.exclude_edge_cells_checkbox)  # Add the checkbox to the top of the left layout

        # Use QGroupBox to group the feature selection area
        feature_group_box = QGroupBox("Select Features for Cell Classification")
        feature_group_layout = QGridLayout()

        self.instruction_label = QLabel()
        feature_layout.addWidget(self.instruction_label)

        self.checkboxes = []
        features = [
            'Cell Area', 'Fitted Ellipse Major Axis', 'Fitted Ellipse Minor Axis', 'Ellipse Major/Minor Axis Ratio', 'Fitted Ellipse Angle', 'Cell Perimeter', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio', 'Approximate Polygon Vertex Count',
            'Circumscribed Circle Radius', 'Inscribed Circle Radius', 'Circularity'
        ]
        # Set spacing between checkboxes
        feature_group_layout.setSpacing(15)  # Set vertical and horizontal spacing between checkboxes

        # Create checkboxes and arrange them in two columns
        for i, feature in enumerate(features):
            checkbox = QCheckBox(feature)
            row = i // 1
            col = i % 1
            feature_group_layout.addWidget(checkbox, row, col)
            self.checkboxes.append(checkbox)

        feature_group_box.setLayout(feature_group_layout)
        feature_layout.addWidget(feature_group_box)

        # Buttons for select all and deselect all
        select_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all)
        select_layout.addWidget(self.select_all_button)

        self.deselect_all_button = QPushButton(" Deselect All")
        self.deselect_all_button.clicked.connect(self.deselect_all)
        select_layout.addWidget(self.deselect_all_button)

        feature_layout.addLayout(select_layout)

        # Confirm button
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        feature_layout.addWidget(self.confirm_button)

        # Set spacing and margins for feature_layout to make it more visually appealing
        feature_layout.setSpacing(10)
        feature_layout.setContentsMargins(20, 20, 20, 20)

        # Create a new horizontal layout for centering alignment
        centered_layout = QHBoxLayout()
        centered_layout.addLayout(feature_layout)
        centered_layout.setAlignment(Qt.AlignCenter)

        # Add the centered layout to the main layout
        main_layout.addLayout(centered_layout)

        # Right: Display images
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(900, 600)  # Set the display size of the image
        main_layout.addWidget(self.image_label)

        # Load the only image in the program's directory
        self.load_image_from_folder()

        # Set the main layout
        self.setLayout(main_layout)

    def load_image_from_folder(self):
        """Load images from the program's directory"""
        folder_path = os.path.join(os.path.dirname(__file__), 'images')  # Assume images are in the 'images' folder in the program directory
        if os.path.exists(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_path = os.path.join(folder_path, image_files[0])  # Assume there is only one image
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=1,
                                              transformMode=Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("No images found")
        else:
            self.image_label.setText("Image folder does not exist")

    def select_all(self):
        """Set all checkboxes to checked"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def deselect_all(self):
        """Set all checkboxes to unchecked"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def get_selected_features(self):
        """Get all selected features"""
        selected_features = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        return selected_features

    def exclude_edge_cells(self):
        """Get the option to exclude edge cells"""
        return self.exclude_edge_cells_checkbox.isChecked()  # Return the state of the checkbox


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setStyleSheet("background-color:white;")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding around the plot

        # Initialize interaction variables
        self.drag_start = None  # Record the starting position of the drag
        self.zoom_factor = 0.9  # Zoom factor
        self.ax = None  # Currently active subplot
        self.connect_events()

    def connect_events(self):
        """Connect mouse events"""
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        """Mouse press event"""
        if event.inaxes:  # Mouse pressed inside the subplot
            self.ax = event.inaxes
            self.drag_start = (event.xdata, event.ydata)

    def on_motion(self, event):
        """Mouse drag event"""
        if self.drag_start is None or not event.inaxes or event.inaxes != self.ax:
            return
        dx = self.drag_start[0] - event.xdata
        dy = self.drag_start[1] - event.ydata

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        self.ax.set_xlim([x_min + dx, x_max + dx])
        self.ax.set_ylim([y_min + dy, y_max + dy])
        self.drag_start = (event.xdata, event.ydata)  # Update the starting point
        self.draw()

    def on_release(self, event):
        """Mouse release event"""
        self.drag_start = None

    def on_scroll(self, event):
        """Mouse scroll event (zoom)"""
        if event.inaxes:
            self.ax = event.inaxes
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2

            scale = self.zoom_factor if event.button == 'up' else 1 / self.zoom_factor
            x_range = (x_max - x_min) * scale
            y_range = (y_max - y_min) * scale

            self.ax.set_xlim([x_mid - x_range / 2, x_mid + x_range / 2])
            self.ax.set_ylim([y_mid - y_range / 2, y_mid + y_range / 2])
            self.draw()



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_main_interface()

    def init_main_interface(self):
        self.setWindowTitle('Epi_Quant')
        self.setFixedSize(1300, 950)  # Fix the window size to 1200x850 pixels

        main_layout = QVBoxLayout()

        # Adjust the margins of the main layout to move the path input area closer to the top
        main_layout.setContentsMargins(10, 0, 10, 10)  # Margins: left, top, right, bottom, set top margin to 5

        # Create the path input area and set a gray background
        path_groupbox = QGroupBox()  # No longer set a title
        path_groupbox.setStyleSheet("""
                QGroupBox {
                    background-color: #DCDCDC;  /* Set background to gray */
                    border: 1px solid lightgray;  /* Set border to light gray */
                    margin-top: 0px;  /* Reduce top margin, set to 0 */
                }
            """)

        path_groupbox.setFixedHeight(120)  # Set fixed height to limit space usage

        path_layout = QVBoxLayout(path_groupbox)  # Directly associate the layout with QGroupBox
        path_layout.setSpacing(0)

        # Set a unified button width
        button_width = 70  # Set button width to 50 pixels

        # Define a fixed width for all input boxes
        input_box_width = 1030  # Set the width of the input box here

        # Original image path
        img_path_hbox = QHBoxLayout()  # Create another horizontal layout container
        self.img_path_label = QLabel('Path to Original Images:')
        self.img_path_input = QLineEdit(self)
        self.img_path_input.setFixedWidth(input_box_width)  # Set fixed width for the input box
        self.img_browse_button = QPushButton("Browse")  # Create a browse button
        self.img_browse_button.setFixedWidth(button_width)  # Set button width
        self.img_browse_button.clicked.connect(self.browse_img_path)  # Bind button click event

        img_path_hbox.addWidget(self.img_path_label)
        img_path_hbox.addStretch()  # Add stretch to push the input box to the right
        img_path_hbox.addWidget(self.img_path_input)  # Add path input box
        img_path_hbox.addWidget(self.img_browse_button)  # Add browse button
        path_layout.addLayout(img_path_hbox)  # Add horizontal layout to vertical layout

        # .npy file path
        npy_path_hbox = QHBoxLayout()  # Create a horizontal layout container
        self.npy_path_label = QLabel('Path to .npy Files:')
        self.npy_path_input = QLineEdit(self)
        self.npy_path_input.setFixedWidth(input_box_width)  # Set fixed width for the input box
        self.npy_browse_button = QPushButton("Browse")  # Create a browse button
        self.npy_browse_button.setFixedWidth(button_width)  # Set button width
        self.npy_browse_button.clicked.connect(self.browse_npy_path)  # Bind button click event

        npy_path_hbox.addWidget(self.npy_path_label)  # Add label
        npy_path_hbox.addStretch()  # Add stretch to push the input box to the right
        npy_path_hbox.addWidget(self.npy_path_input)  # Add path input box
        npy_path_hbox.addWidget(self.npy_browse_button)  # Add browse button
        path_layout.addLayout(npy_path_hbox)  # Add horizontal layout to vertical layout

        # Output folder path
        output_path_hbox = QHBoxLayout()  # Create another horizontal layout container
        self.output_path_label = QLabel('Output Folder Path:')
        self.output_path_input = QLineEdit(self)
        self.output_path_input.setFixedWidth(input_box_width)  # Set fixed width for the input box
        self.output_browse_button = QPushButton("Browse")  # Create a browse button
        self.output_browse_button.setFixedWidth(button_width)  # Set button width
        self.output_browse_button.clicked.connect(self.browse_output_path)  # Bind button click event

        output_path_hbox.addWidget(self.output_path_label)
        output_path_hbox.addStretch()  # Add stretch to push the input box to the right
        output_path_hbox.addWidget(self.output_path_input)  # Add path input box
        output_path_hbox.addWidget(self.output_browse_button)  # Add browse button
        path_layout.addLayout(output_path_hbox)  # Add horizontal layout to vertical layout

        # Add the path input area to the main layout
        main_layout.addWidget(path_groupbox)

        # Bind text change events of input path boxes
        self.npy_path_input.textChanged.connect(self.update_paths_and_create_folders)
        self.img_path_input.textChanged.connect(self.update_paths_and_create_folders)
        self.output_path_input.textChanged.connect(self.update_paths_and_create_folders)
        # Create image display and function button area
        content_layout = QHBoxLayout()

        # Left area, including buttons and image display area
        left_layout = QVBoxLayout()

        # Use QTabWidget instead of three buttons
        self.tab_widget = QTabWidget()

        # Set the width of tab_widget to fill the layout
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tab_widget.setSizePolicy(size_policy)

        # Create five tabs
        self.original_image_tab = QWidget()
        self.classification_image_tab = QWidget()
        self.tracking_image_tab = QWidget()

        # Set styles for QTabWidget and QTabBar to hide the gray background bar
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {  /* Hide the background below the tabs */
                border: 0px;
                background: transparent;
            }
            QTabBar::tab {
                background-color: white;  /* Set tab background color */
                border: 1px solid lightgray;  /* Set tab border */
                padding: 3px;
                min-width: 315px;  /* Set minimum tab width to 250 pixels */
                border-radius: 0px;  /* Set border radius to 10 pixels */
            }
            QTabBar::tab:selected {  /* Set style for selected tab */
                background-color: blue;  /* Make the entire tab background blue */
                color: white;  /* Set text color to white when selected for visibility */
                border: 2px solid blue;  /* Set the entire tab border color to blue */
            }
            QTabBar::tab:disabled {  /* Style for disabled tabs */
                background-color: #D3D3D3;  /* Gray background when disabled */
                color: darkgray;  /* Dark gray text when disabled */
                border: 1px solid  #D3D3D3;  /* Gray border when disabled */
    }
        """)

        # Add tabs to QTabWidget
        self.tab_widget.addTab(self.original_image_tab, "Original Cell Images")
        self.tab_widget.addTab(self.classification_image_tab, "Cell Classification Images")
        self.tab_widget.addTab(self.tracking_image_tab, "Cell Tracking Images")

        # Disable tabs
        self.tab_widget.setTabEnabled(0, False)  # Disable "Original Cell Images" tab
        self.tab_widget.setTabEnabled(1, False)  # Disable "Cell Classification Images" tab
        self.tab_widget.setTabEnabled(2, False)  # Disable "Cell Tracking Images" tab

        # Connect tab switch signal to image loading function
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Add tabs to layout, remove spacing and padding
        left_layout.addWidget(self.tab_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        left_layout.setSpacing(5)  # Reduce spacing between components

        # Image display area
        self.image_layout = QVBoxLayout()
        self.canvas_widget = QWidget()
        self.canvas_widget.setLayout(self.image_layout)

        self.canvas_widget.setStyleSheet("""
            background-color: #DCDCDC;  /* Set to gray-white */
        """)

        # Place canvas_widget at the top
        left_layout.addWidget(self.canvas_widget)

        # Add vertical scrollbar
        self.scroll_bar = QScrollBar(Qt.Vertical)
        self.scroll_bar.setMinimum(1)  # Minimum value of scrollbar
        self.scroll_bar.valueChanged.connect(self.scroll_value_changed)  # Connect scrollbar value change signal
        self.scroll_bar.setEnabled(False)  # Initially disabled, enable after loading images

        # Layout image display area and scrollbar in content area
        content_layout.addLayout(left_layout, 3)
        content_layout.addWidget(self.scroll_bar, 1)  # Place scrollbar on the right

        # Create feature_layout for right-side function buttons
        feature_layout = QVBoxLayout()

        # Define button styles as class attributes
        self.button_style_disabled = """
                QPushButton {
                    background-color: #D3D3D3;   /* Gray-white background */
                    color: darkgray;              /* Text color */
                    border: 1px solid gray;       /* Border color and thickness */
                    border-radius: 5px;           /* Set to 0 to ensure no rounded corners */
                    font-size: 14px;              /* Set text font size */
                }
            """

        self.button_style_enabled = """
                QPushButton {
                    background-color: white;      /* Background color */
                    color: black;                 /* Text color */
                    border: 1px solid black;      /* Border color and thickness */
                    border-radius: 5px;           /* Set to 0 to ensure no rounded corners */
                    font-size: 14px;              /* Set text font size */
                }
            """

        # **Work Area Buttons**
        work_area_group = QGroupBox("Work Area")
        work_area_layout = QVBoxLayout()
        work_area_group.setFixedHeight(300)  # Adjust height to 300 pixels

        # Unified button spacing and margins
        work_area_layout.setSpacing(10)
        work_area_layout.setContentsMargins(10, 10, 10, 10)  # Margins: left, top, right, bottom

        self.feature_extraction_button = QPushButton("Extract Cell Features")
        self.feature_extraction_button.setStyleSheet(self.button_style_disabled)
        self.feature_extraction_button.setFixedSize(250, 40)
        self.feature_extraction_button.setEnabled(False)
        work_area_layout.addWidget(self.feature_extraction_button, alignment=Qt.AlignCenter)
        self.feature_extraction_button.clicked.connect(self.select_features_and_extract_info)

        self.classification_button = QPushButton("Classify Cells")
        self.classification_button.setStyleSheet(self.button_style_disabled)
        self.classification_button.setFixedSize(250, 40)
        self.classification_button.setEnabled(False)
        work_area_layout.addWidget(self.classification_button, alignment=Qt.AlignCenter)
        self.classification_button.clicked.connect(self.perform_combined_classification)

        self.correction_button = QPushButton("Correct Cell Classification")
        self.correction_button.setStyleSheet(self.button_style_disabled)
        self.correction_button.setFixedSize(250, 40)
        self.correction_button.setEnabled(False)
        work_area_layout.addWidget(self.correction_button, alignment=Qt.AlignCenter)
        self.correction_button.clicked.connect(self.show_correction_window)

        self.cell_tracking_button = QPushButton("Track Cells")
        self.cell_tracking_button.setStyleSheet(self.button_style_disabled)
        self.cell_tracking_button.setFixedSize(250, 40)
        self.cell_tracking_button.setEnabled(False)
        work_area_layout.addWidget(self.cell_tracking_button, alignment=Qt.AlignCenter)
        self.cell_tracking_button.clicked.connect(self.perform_cell_tracking_and_generate_images)

        self.tracking_correction_button = QPushButton("Correct Cell Tracking")
        self.tracking_correction_button.setStyleSheet(self.button_style_disabled)
        self.tracking_correction_button.setFixedSize(250, 40)
        self.tracking_correction_button.setEnabled(False)
        work_area_layout.addWidget(self.tracking_correction_button, alignment=Qt.AlignCenter)
        self.tracking_correction_button.clicked.connect(self.show_tracking_correction_input)

        self.generate_data_button = QPushButton("Generate Quantitative Data")
        self.generate_data_button.setStyleSheet(self.button_style_disabled)
        self.generate_data_button.setFixedSize(250, 40)
        self.generate_data_button.setEnabled(False)
        work_area_layout.addWidget(self.generate_data_button, alignment=Qt.AlignCenter)
        self.generate_data_button.clicked.connect(self.generate_and_notify)

        # Add work area layout to QGroupBox
        work_area_group.setLayout(work_area_layout)

        # **Display Area Buttons**
        display_area_group = QGroupBox("Display Area")
        display_area_layout = QVBoxLayout()
        display_area_group.setFixedHeight(300)  # Adjust height to 300 pixels

        # Unified button spacing and margins
        display_area_layout.setSpacing(10)
        display_area_layout.setContentsMargins(10, 10, 10, 10)  # Margins: left, top, right, bottom

        self.report_button = QPushButton("Cell Classification Report")
        self.report_button.setStyleSheet(self.button_style_disabled)
        self.report_button.setFixedSize(250, 40)
        self.report_button.setEnabled(False)
        display_area_layout.addWidget(self.report_button, alignment=Qt.AlignCenter)
        self.report_button.clicked.connect(self.show_report_window)

        self.direction_button = QPushButton("Display Cell Displacement")
        self.direction_button.setStyleSheet(self.button_style_disabled)
        self.direction_button.setFixedSize(250, 40)
        self.direction_button.setEnabled(False)
        display_area_layout.addWidget(self.direction_button, alignment=Qt.AlignCenter)
        self.direction_button.clicked.connect(self.show_celldireciton_window)

        self.track_button = QPushButton("Show Cell Tracking Trajectories")
        self.track_button.setStyleSheet(self.button_style_disabled)
        self.track_button.setFixedSize(250, 40)
        self.track_button.setEnabled(False)
        display_area_layout.addWidget(self.track_button, alignment=Qt.AlignCenter)
        self.track_button.clicked.connect(self.show_plot_track_window)

        self.show_analysis_button = QPushButton("Single-Cell Quantitative Analysis")
        self.show_analysis_button.setStyleSheet(self.button_style_disabled)
        self.show_analysis_button.setFixedSize(250, 40)
        self.show_analysis_button.setEnabled(False)
        display_area_layout.addWidget(self.show_analysis_button, alignment=Qt.AlignCenter)
        self.show_analysis_button.clicked.connect(self.show_quantitative_analysis)

        self.cell_shadow_plot = QPushButton("Multi-Cell Quantitative Analysis")
        self.cell_shadow_plot.setStyleSheet(self.button_style_disabled)
        self.cell_shadow_plot.setFixedSize(250, 40)
        self.cell_shadow_plot.setEnabled(False)
        display_area_layout.addWidget(self.cell_shadow_plot, alignment=Qt.AlignCenter)
        self.cell_shadow_plot.clicked.connect(self.show_cell_shadow_plot)

        # Add display area layout to QGroupBox
        display_area_group.setLayout(display_area_layout)

        # **Operation Log Area**
        operations_log_group = QGroupBox("Operation Log")
        operations_log_layout = QVBoxLayout()
        operations_log_group.setFixedHeight(100)  # Adjust height to 100 pixels

        # Use QTextEdit for operation log
        self.operations_log = QTextEdit()
        self.operations_log.setReadOnly(True)  # Set to read-only
        self.operations_log.setEnabled(False)  # Disable operation log

        operations_log_layout.addWidget(self.operations_log)
        operations_log_group.setLayout(operations_log_layout)

        # **Add work area, display area, and operation log area to feature_layout**
        feature_layout.addWidget(work_area_group)  # Add work area
        feature_layout.addWidget(display_area_group)  # Add display area

        # Set layout to window
        content_layout.addLayout(feature_layout, 1)

        self.canvas_widget.setFixedHeight(self.canvas_widget.height() + 185)

        main_layout.addLayout(content_layout)

        main_layout.addWidget(operations_log_group)  # Add operation log area at the bottom

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Disable all buttons after initialization
        # self.disable_buttons()

    def update_paths_and_create_folders(self):
        # Get and update paths
        npy_path = self.npy_path_input.text().strip()
        img_path = self.img_path_input.text().strip()
        output_path = self.output_path_input.text().strip()

        # Update paths in the config object
        config.npy_folder_path = npy_path
        config.Img_path = img_path
        config.output_path = output_path
        config.cell_classification_output_path = os.path.join(output_path, 'cell_classification_output')
        config.cell_track_output_path = os.path.join(output_path, 'cell_track_output')
        config.quantitative_analysis_output_path = os.path.join(output_path, 'quantitative_analysis_output')

        # Check if all paths are entered
        if not (npy_path and img_path and output_path):
            self.disable_buttons()  # Disable buttons if paths are not fully entered
            self.tab_widget.setEnabled(False)
            return

        try:
            # Attempt to create output directories and subdirectories
            if not os.path.exists(output_path):
                #print(f"Attempting to create directory: {output_path}")
                os.makedirs(output_path)

            if not os.path.exists(config.cell_classification_output_path):
                #print(f"Attempting to create directory: {config.cell_classification_output_path}")
                os.makedirs(config.cell_classification_output_path)

            if not os.path.exists(config.cell_track_output_path):
                #print(f"Attempting to create directory: {config.cell_track_output_path}")
                os.makedirs(config.cell_track_output_path)

            if not os.path.exists(config.quantitative_analysis_output_path):
                #print(f"Attempting to create directory: {config.quantitative_analysis_output_path}")
                os.makedirs(config.quantitative_analysis_output_path)

            self.tab_widget.setTabEnabled(0, True)  # Enable "Original Cell Images" tab
            self.tab_widget.setTabEnabled(1, True)  # Enable "Cell Classification Images" tab
            self.tab_widget.setTabEnabled(2, True)  # Enable "Cell Tracking Images" tab

            # Enable tabs and functional buttons
            self.tab_widget.setEnabled(True)
            self.feature_extraction_button.setEnabled(True)
            self.feature_extraction_button.setStyleSheet(self.button_style_enabled)

            self.tab_widget.setEnabled(True)
            self.classification_button.setEnabled(True)
            self.classification_button.setStyleSheet(self.button_style_enabled)

            self.report_button.setEnabled(True)
            self.report_button.setStyleSheet(self.button_style_enabled)

            self.direction_button.setEnabled(True)
            self.direction_button.setStyleSheet(self.button_style_enabled)

            self.correction_button.setEnabled(True)
            self.correction_button.setStyleSheet(self.button_style_enabled)

            self.cell_tracking_button.setEnabled(True)
            self.cell_tracking_button.setStyleSheet(self.button_style_enabled)

            self.tracking_correction_button.setEnabled(True)
            self.tracking_correction_button.setStyleSheet(self.button_style_enabled)

            self.track_button.setEnabled(True)
            self.track_button.setStyleSheet(self.button_style_enabled)

            self.generate_data_button.setEnabled(True)
            self.generate_data_button.setStyleSheet(self.button_style_enabled)

            self.show_analysis_button.setEnabled(True)
            self.show_analysis_button.setStyleSheet(self.button_style_enabled)

            self.cell_shadow_plot.setEnabled(True)
            self.cell_shadow_plot.setStyleSheet(self.button_style_enabled)

            # Actively load original cell images
            self.load_images(config.Img_path, image_type="original")

            # Check and disable "Cell Classification Images" and "Cell Tracking Images" tabs
            self.check_folder_and_toggle_tab(
                os.path.join(config.cell_classification_output_path, 'cells_clustering_results_pictures'), 1)
            self.check_folder_and_toggle_tab(os.path.join(config.cell_track_output_path, 'cell_track_output_pictures'),
                                             2)

            # Ensure the first tab (Original Cell Images) is selected by default during initialization
            self.tab_widget.setCurrentIndex(0)

            # Enable operation log
            self.operations_log.setEnabled(True)

        except OSError as e:
            # Catch folder creation errors and display detailed error messages
            QMessageBox.warning(self, "Error", f"Folder creation failed: {e.strerror}")

    def log_operation(self, operation_text, start_time=None):
        """
        Record and display executed operations, including start and end times.
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if start_time:
            # Record operation end time
            operation_message = f"Completed: {operation_text} at {current_time}"
        else:
            # Record operation start time
            operation_message = f"Started: {operation_text} at {current_time}"

        self.operations_log.append(operation_message)
        self.operations_log.repaint()  # Force refresh of the operation log display

    def show_report_window(self):
        analysis_data_dir = os.path.join(config.quantitative_analysis_output_path,
                                         'all_cell_quantitative_analysis_output')
        if not os.path.exists(analysis_data_dir) or not os.listdir(analysis_data_dir):
            QMessageBox.warning(self, "Notice", "Quantitative analysis data not found. Please generate it first.")
            return
        self.log_operation("Cell Classification Report")
        report_dialog = ReportDialog(self)
        report_dialog.exec_()

    def show_celldireciton_window(self):
        analysis_data_dir = os.path.join(config.quantitative_analysis_output_path,
                                         'all_cell_quantitative_analysis_output')
        if not os.path.exists(analysis_data_dir) or not os.listdir(analysis_data_dir):
            QMessageBox.warning(self, "Notice", "Quantitative analysis data not found. Please generate it first.")
            return
        self.log_operation("Display Cell Displacement")
        start_time = time.time()

        self.celldirection_window = CellDirection(config.Img_path, config.npy_folder_path, config.cell_track_output_path, config.cell_classification_output_path)

        self.celldirection_window.show()
        self.log_operation("Display Cell Displacement", start_time=start_time)

    def browse_npy_path(self, event):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for .npy Files")
        if dir_path:
            self.npy_path_input.setText(dir_path)

    def browse_img_path(self, event):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Original Images")
        if dir_path:
            self.img_path_input.setText(dir_path)
            self.load_images(dir_path, image_type="original")

    def browse_output_path(self, event):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path_input.setText(dir_path)

    def check_folder_and_toggle_tab(self, folder_path, tab_index):
        if os.path.exists(folder_path) and os.listdir(folder_path):  # Ensure the folder exists and is not empty
            self.tab_widget.setTabEnabled(tab_index, True)
        else:
            self.tab_widget.setTabEnabled(tab_index, False)

    def on_tab_changed(self, index):
        original_cells_path = self.img_path_input.text()

        # Get the latest path for cell classification images
        classification_images_path = os.path.join(config.cell_classification_output_path,
                                                  'cells_clustering_results_pictures')
        tracking_images_path = os.path.join(config.cell_track_output_path,
                                            'cell_track_output_pictures')

        # Disable or enable tabs based on folder existence
        self.check_folder_and_toggle_tab(original_cells_path, 0)
        self.check_folder_and_toggle_tab(classification_images_path, 1)
        self.check_folder_and_toggle_tab(tracking_images_path, 2)

        if index == 0:
            self.load_images(original_cells_path, image_type="original")
        elif index == 1:
            self.load_images(classification_images_path, image_type="classification")
        elif index == 2:
            self.load_images(tracking_images_path)

    def perform_cell_tracking_and_generate_images(self):
        start_time = time.time()
        self.log_operation("Track Cells", start_time=None)
        self.show_progress_bar("Tracking cells in progress, please wait...")
        self.progress_dialog.setValue(1)

        output_path = config.cell_track_output_path
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        self.progress_dialog.setValue(5)
        npy_folder = config.npy_folder_path
        img_folder = config.Img_path
        all_cells_tracking_output_path = None
        cells_info_path = os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx')

        if not os.path.exists(cells_info_path):
            self.progress_dialog.close()  # Close the progress bar

            QMessageBox.warning(self, "Notice", "Cells_info.xlsx not found. Please extract cell features first.")
            return

        cells_info = pd.read_excel(cells_info_path)

        # Check if the 'leading_edge' column exists to determine if there are two cell types
        has_leading_edge = 'leading_edge' in cells_info.columns
        num_cell_types = 2 if has_leading_edge else 1

        if has_leading_edge:
            self.show_progress_bar("Starting classified cell tracking...")

            # Divide cells into large cells and small cells
            big_cells_info = cells_info[cells_info['leading_edge'] < 0].copy()
            big_cells_info_path = os.path.join(config.cell_track_output_path, 'mes_cells_info.xlsx')
            big_cells_info.to_excel(big_cells_info_path, sheet_name='mesCells', index=False)

            small_cells_info = cells_info[cells_info['leading_edge'] > 0].copy()
            small_cells_info_path = os.path.join(config.cell_track_output_path, 'epi_cells_info.xlsx')
            small_cells_info.to_excel(small_cells_info_path, sheet_name='epiCells', index=False)

            big_cells_info.set_index('Cell Index', inplace=True)
            small_cells_info.set_index('Cell Index', inplace=True)

            all_tracking_data = pd.DataFrame()

            # Define progress bar segments
            # Assume the cell tracking part accounts for 50% of the total progress, with large and small cells each taking 25%
            tracking_start = 5
            tracking_end = 50
            half_progress = (tracking_end - tracking_start) / 2  # Progress allocated to each cell type (22.5%)

            # Process large cells
            if not big_cells_info.empty:
                big_tracking_data = self.run_cell_tracking(
                    big_cells_info,
                    progress_start=tracking_start,
                    progress_end=tracking_start + half_progress
                )
                all_tracking_data = pd.concat([all_tracking_data, big_tracking_data], axis=0)

            # Process small cells
            if not small_cells_info.empty:
                small_tracking_data = self.run_cell_tracking(
                    small_cells_info,
                    progress_start=tracking_start + half_progress,
                    progress_end=tracking_end
                )
                all_tracking_data = pd.concat([all_tracking_data, small_tracking_data], axis=0)

        else:
            self.show_progress_bar("Starting unclassified cell tracking...")
            cells_info.set_index('Cell Index', inplace=True)
            all_tracking_data = self.run_cell_tracking(
                cells_info,
                progress_start=5,
                progress_end=50
            )

        # Save all tracking data
        if all_tracking_data is not None and not all_tracking_data.empty:
            all_cells_tracking_output_folder = os.path.join(output_path, 'all_cell_tracking')
            all_cells_tracking_output_path = os.path.join(all_cells_tracking_output_folder,
                                                          'all_cell_merged_tracking_results.xlsx')
            os.makedirs(all_cells_tracking_output_folder, exist_ok=True)
            all_tracking_data.to_excel(all_cells_tracking_output_path, index=False)

        # Update progress bar to 50% (cell tracking part completed)
        if has_leading_edge:
            self.progress_dialog.setValue(tracking_end)  # 50%
        else:
            self.progress_dialog.setValue(50)

        # Generate comparison images
        if all_cells_tracking_output_path:
            all_comparison_window = ComparisonMainWindow(
                npy_folder,
                img_folder,
                all_cells_tracking_output_path,
                'all_cells'
            )
            # Image generation part accounts for the remaining progress (50% to 100%)
            all_comparison_window.preprocess_images(
                self.progress_dialog,
                self.show_progress_bar,
                initial_value=50
            )
            QMessageBox.information(self, 'Completed',
                                    'Cell tracking and image generation have been completed and saved.')
        else:
            QMessageBox.warning(self, 'Error',
                                'Unable to generate tracking images because the tracking file does not exist.')

        self.progress_dialog.close()

        self.check_folder_and_toggle_tab(config.cell_track_output_path, 2)
        self.log_operation("Track Cells", start_time=start_time)

    def select_features_and_extract_info(self):

        start_time = time.time()
        self.log_operation("Extract Cell Features", start_time=None)

        dialog = FeatureSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_features = dialog.get_selected_features()
            self.exclude_edge_cells_option = dialog.exclude_edge_cells()

            if not self.selected_features:
                QMessageBox.warning(self, "Warning", "Please select at least one feature.")
                return False

            try:
                self.show_progress_bar("Calculating cell features, please wait...")
                self.progress_dialog.setValue(5)
                self.cells_info, fig_num = self.bio_extract_cell_info(
                    config.npy_folder_path,
                    config.cell_classification_output_path,
                    exclude_edge_cells=self.exclude_edge_cells_option
                )
                self.show_progress_bar("Cell feature calculation completed.")
                self.progress_dialog.setValue(100)
                self.progress_dialog.close()
                self.log_operation("Extracting cell features", start_time=start_time)

                # Notify extraction completion
                QMessageBox.information(self, "Completed", "Cell feature extraction has been completed.")

                return True

                # Enable other functional buttons
                self.classification_button.setEnabled(True)
                self.classification_button.setStyleSheet(self.button_style_enabled)
                self.correction_button.setEnabled(True)
                self.correction_button.setStyleSheet(self.button_style_enabled)
                self.cell_tracking_button.setEnabled(True)
                self.cell_tracking_button.setStyleSheet(self.button_style_enabled)
                self.tracking_correction_button.setEnabled(True)
                self.tracking_correction_button.setStyleSheet(self.button_style_enabled)
                self.generate_data_button.setEnabled(True)
                self.generate_data_button.setStyleSheet(self.button_style_enabled)
                self.report_button.setEnabled(True)
                self.report_button.setStyleSheet(self.button_style_enabled)
                self.direction_button.setEnabled(True)
                self.direction_button.setStyleSheet(self.button_style_enabled)
                self.track_button.setEnabled(True)
                self.track_button.setStyleSheet(self.button_style_enabled)
                self.show_analysis_button.setEnabled(True)
                self.show_analysis_button.setStyleSheet(self.button_style_enabled)
                self.cell_shadow_plot.setEnabled(True)
                self.cell_shadow_plot.setStyleSheet(self.button_style_enabled)


            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cell feature extraction failed: {str(e)}")
                self.log_operation("Cell feature extraction failed")
        else:
            QMessageBox.warning(self, "Operation Cancelled", "Cell feature extraction operation cancelled.")
            self.log_operation("Feature selection cancelled by user")
            return False

    def is_edge_cell(self, cell_pixels, height, width, margin):
        """Determine whether a cell is located at or near the edge of the image within a certain margin."""
        min_x = np.min(cell_pixels[1])
        max_x = np.max(cell_pixels[1])
        min_y = np.min(cell_pixels[0])
        max_y = np.max(cell_pixels[0])

        # Check if the cell contour is within the margin range of the image boundary
        if (min_x <= margin or max_x >= width - margin or min_y <= margin or max_y >= height - margin):
            return True
        return False

    def bio_extract_cell_info(self, path, output_path, exclude_edge_cells=False):

        cells_info_columns = [
            'Cell Area', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio',
            'Cell Perimeter', 'Approximate Polygon Vertex Count',
            'Fitted Ellipse Minor Axis', 'Fitted Ellipse Major Axis', 'Ellipse Major/Minor Axis Ratio',
            'Fitted Ellipse Angle', 'Circumscribed Circle Radius',
            'Inscribed Circle Radius', 'Center X Coordinate', 'Center Y Coordinate', 'Cell Left Boundary',
            'Cell Right Boundary', 'Circularity', 'P-Value'
        ]

        cells_info = pd.DataFrame(columns=cells_info_columns)
        fig_num = 0
        # Get all files to process
        npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]

        # Extract numerical parts and sort starting from 1
        def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else float('inf')  # Files without numbers go to the end

        npy_files = sorted(npy_files, key=extract_number)

        # Optionally, filter out files that do not start from 1 or have gaps
        npy_files = [f for f in npy_files if extract_number(f) >= 1]

        total_files = len(npy_files)

        for idx, filename in enumerate(npy_files):
            basename, ext = os.path.splitext(filename)
            if ext != '.npy':
                continue
            dat = np.load(os.path.join(path, filename), allow_pickle=True).item()
            outlines = dat['outlines']  # Extract outlines data
            height, width = outlines.shape  # Get the height and width of the image
            fig_num += 1
            cells_info_fig = pd.DataFrame(columns=cells_info_columns)  # Create an empty cells_info to store information
            regex = re.compile(r'\d+')
            fig_id = str(max(map(int, regex.findall(filename))))  # Calculate the image number
            frame_number = int(fig_id) if fig_id else idx + 1

            self.show_progress_bar(f'Extracting cell features for frame {frame_number}...')
            progress_value = int((idx + 1) / total_files * 100)  # Calculate progress
            self.progress_dialog.setValue(progress_value)  # Update progress bar value
            QApplication.processEvents()  # Force refresh of the event loop

            unique_ids = np.unique(outlines)  # Extract all unique IDs
            unique_ids = unique_ids[unique_ids != 0]  # Exclude background ID 0

            for cell_id in unique_ids:
                positions = np.where(outlines == cell_id)

                # Decide whether to exclude edge cells based on the passed parameter
                if exclude_edge_cells:  # If the option to exclude edge cells is selected
                    if self.is_edge_cell(positions, height, width, margin=3):
                        continue  # Skip if it is an edge cell

                contour_points = np.array([positions[1], positions[0]]).T.reshape(-1, 1, 2)

                # === Use contour points for area, perimeter, and polygon fitting ===
                fill_img = np.zeros_like(outlines, dtype=np.uint8)
                cv2.fillPoly(fill_img, [contour_points], 1)
                cell_area = np.sum(fill_img)

                # Extract contours
                contours, _ = cv2.findContours(fill_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    filled_contour = contours[0]
                    cell_arclength = int(cv2.arcLength(filled_contour, True))
                    vertices = cv2.approxPolyDP(filled_contour, 2, True)

                    # Calculate the minimum enclosing circle and maximum inscribed circle
                    _, radius = cv2.minEnclosingCircle(filled_contour)
                    min_enclosing_circle_radius = int(radius)

                    dist_transform = cv2.distanceTransform(fill_img, cv2.DIST_L2, 5)
                    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
                    max_inner_circle_radius = int(max_val)

                    # Check if cell_arclength is 0 to avoid division by 0
                    if cell_arclength > 0:
                        circularity = (4 * math.pi * cell_area) / (cell_arclength ** 2)
                        p_value = (4 * math.pi) / (circularity ** 2)
                    else:
                        circularity = -1  # Or set to another reasonable value
                        p_value = -1
                else:
                    vertices = []
                    min_enclosing_circle_radius = 0
                    max_inner_circle_radius = 0

                # === Ellipse fitting ===
                if len(contour_points) >= 5:
                    cell_ellipse = cv2.fitEllipse(contour_points)
                    ratio_long_and_short_axes = cell_ellipse[1][1] / cell_ellipse[1][0] if cell_ellipse[1][
                                                                                               0] != 0 else -1
                    cell_ellipse_angle = 180 - cell_ellipse[2] if cell_ellipse[2] > 90 else cell_ellipse[2]
                else:
                    ratio_long_and_short_axes = -1
                    cell_ellipse_angle = -1

                # Calculate the center point (cx, cy)
                cy = int(np.mean(positions[0]))
                cx = int(np.mean(positions[1]))

                # Calculate the value of AP/DV
                ap = np.max(positions[1]) - np.min(positions[1])
                dv = np.max(positions[0]) - np.min(positions[0])
                aspect_ratio = ap / dv if dv != 0 else -1

                # Save information
                cell_info = [
                    cell_area, ap, dv, aspect_ratio, cell_arclength, len(vertices),
                    cell_ellipse[1][0] if len(contour_points) >= 5 else -1,
                    cell_ellipse[1][1] if len(contour_points) >= 5 else -1,
                    ratio_long_and_short_axes, cell_ellipse_angle,
                    min_enclosing_circle_radius, max_inner_circle_radius,
                    cx, cy, np.min(positions[1]), np.max(positions[1]), circularity, p_value
                ]
                cell_info = np.round(cell_info, 2)
                cell_index = f'Cell{fig_id}_{cell_id}'
                cell_info_df = pd.DataFrame([cell_info], index=[cell_index], columns=cells_info_columns)
                cells_info_fig = pd.concat([cells_info_fig, cell_info_df])

            cells_info = pd.concat([cells_info, cells_info_fig])

        # Filter out rows containing -1
        # cells_info = cells_info[~(cells_info == -1).any(axis=1)]
        print(cells_info[(cells_info == -1).any(axis=1)])
        # Export "Cell Index" as the first column to an Excel file
        cells_info.index.name = 'Cell Index'
        cells_info.to_excel(os.path.join(output_path, 'Cells_info.xlsx'), sheet_name='Cell', index=True)

        self.show_progress_bar(f'Cell feature extraction for frame {frame_number} completed...')

        return cells_info, fig_num

    def perform_combined_classification(self):

        if not hasattr(self, 'cells_info') or self.cells_info.empty:
            QMessageBox.warning(self, "Notice", "Please extract cell features before performing this operation.")
            return

        start_time = time.time()

        self.log_operation("Classify Cells", start_time=None)
        self.show_progress_bar("Performing PCA and clustering analysis, please wait...")

        try:
            self.progress_dialog.setValue(5)
            cells_info_filtered = self.cells_info.loc[:, self.selected_features]

            cells_info_pca = self.bio_compute_cell_pca(cells_info_filtered, config.cell_classification_output_path)

            self.progress_dialog.setValue(10)
            self.bio_clustering_k_mean(cells_info_pca, config.cell_classification_output_path)

            self.show_progress_bar("PCA and clustering completed...")
            self.progress_dialog.setValue(15)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"PCA or clustering analysis failed: {str(e)}")
        try:
            processor = CellClassification(
                npy_folder=config.npy_folder_path,
                img_folder=config.Img_path,
                output_picture=os.path.join(config.cell_classification_output_path,
                                            'cells_clustering_results_pictures'),
                clustering_data_path=os.path.join(config.cell_classification_output_path,
                                                  'cells_clustering_results.xlsx'),
                cells_info_path=os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx'),
                output_folder=config.cell_classification_output_path,
            )
            processor.process_images(self.progress_dialog, self.show_progress_bar, initial_value=30)
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()
            self.log_operation("Classify Cells", start_time=start_time)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Image processing failed: {str(e)}")
            self.log_operation("Image processing failed")
            return

        QMessageBox.information(self, "Completed", "Cell classification images have been generated and saved.")
        self.check_folder_and_toggle_tab(config.cell_classification_output_path, 1)

    def show_progress_bar(self, text):

        if not hasattr(self, 'progress_dialog'):
            self.progress_dialog = QProgressDialog(text, None, 0, 100, self)
            self.progress_dialog.setWindowTitle("Progress")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()


            label = self.progress_dialog.findChild(QLabel)
            if label:
                label.setFixedWidth(300)
                label.setWordWrap(True)
        else:
            self.progress_dialog.setLabelText(text)

    def disable_buttons(self):

        self.tab_widget.setEnabled(False)

        self.classification_button.setEnabled(False)
        # self.show_report_window.setEnabled(False)
        self.correction_button.setEnabled(False)
        self.cell_tracking_button.setEnabled(False)
        self.tracking_correction_button.setEnabled(False)
        self.track_button.setEnabled(False)
        self.generate_data_button.setEnabled(False)
        self.show_analysis_button.setEnabled(False)

    def show_quantitative_analysis(self):

        analysis_data_dir = os.path.join(config.quantitative_analysis_output_path,
                                         'all_cell_quantitative_analysis_output')
        if not os.path.exists(analysis_data_dir) or not os.listdir(analysis_data_dir):
            QMessageBox.warning(self, "Notice", "Quantitative analysis data not found. Please generate it first.")
            return

        # Record start time
        start_time = time.time()

        npy_folder = config.npy_folder_path

        # Ensure the path exists
        if os.path.exists(npy_folder):
            npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
            npy_file_count = len(npy_files)
        else:
            print(f"The directory {npy_folder} does not exist. Please check the path.")


        self.log_operation("Displaying quantitative analysis results", start_time=None)

        self.analysis_window = QuantitativeAnalysisGUI(npy_file_count)
        self.analysis_window.show()
        self.log_operation("Displaying quantitative analysis results", start_time=start_time)

    def show_cell_shadow_plot(self):

        analysis_data_dir = os.path.join(config.quantitative_analysis_output_path,
                                         'all_cell_quantitative_analysis_output')
        if not os.path.exists(analysis_data_dir) or not os.listdir(analysis_data_dir):
            QMessageBox.warning(self, "Notice", "Quantitative analysis data not found. Please generate it first.")
            return

        start_time = time.time()
        all_cell_quantitative_analysis_output = os.path.join(config.quantitative_analysis_output_path, 'all_cell_quantitative_analysis_output')
        cell_track_output_pictures = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')

        self.log_operation("Displaying quantitative analysis results", start_time=None)
        self.analysis_window = CellShadowLineGUI(config.quantitative_analysis_output_path, all_cell_quantitative_analysis_output, cell_track_output_pictures)
        self.analysis_window.show()
        self.log_operation("Displaying quantitative analysis results", start_time=start_time)

    def show_tracking_correction_input(self):

        start_time = time.time()

        self.log_operation("Correct Cell Tracking", start_time=None)

        tracking_data_path = os.path.join(config.cell_track_output_path, 'all_cell_tracking',
                                            'all_cell_merged_tracking_results.xlsx')
        if not os.path.exists(tracking_data_path):
            QMessageBox.warning(self, "Notice", "Cell tracking results not found. Please perform cell tracking first.")
            return


        cell_type = 'all_cells'



        # Directly open the correction window and pass the file path and cell type (no longer pass the frame number)
        self.open_tracking_correction_window(tracking_data_path, cell_type)

        # Check the folder and update the interface
        self.check_folder_and_toggle_tab(config.cell_track_output_path, 2)

        self.log_operation("Correct Cell Tracking", start_time=start_time)

    def open_tracking_correction_window(self, tracking_data_path, cell_type):
        npy_folder = config.npy_folder_path  # Get the npy folder path from the configuration
        img_folder = config.Img_path

        # Instantiate ComparisonMainWindow and pass the cell_type
        self.comparison_window = ComparisonMainWindow(npy_folder, img_folder,  tracking_data_path, cell_type)

        # Directly display the correction window
        self.comparison_window.show()

    def show_correction_window(self):
        # Record the start time
        start_time = time.time()

        # Log the start of cell classification correction
        self.log_operation("Correct Cell Classification", start_time=None)

        clustering_result_path = os.path.join(config.cell_classification_output_path, 'cells_clustering_results.xlsx')
        cells_info_path = os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx')
        if not os.path.exists(clustering_result_path) or not os.path.exists(cells_info_path):
            QMessageBox.warning(self, "Notice", "Cell classification results not found. Please perform cell classification first.")
            return


        # Open the CellClassification window and jump to the specified frame
        self.correction_window = CellClassification(
            npy_folder=config.npy_folder_path,
            img_folder=config.Img_path,
            output_picture=os.path.join(config.cell_classification_output_path,
                                        'cells_clustering_results_pictures'),
            clustering_data_path=os.path.join(config.cell_classification_output_path,
                                                'cells_clustering_results.xlsx'),
            cells_info_path=os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx'),
            output_folder=config.cell_classification_output_path,
        )
        self.correction_window.show()

        # Use the existing function to enable the "Cell Classification Images" tab
        self.check_folder_and_toggle_tab(config.cell_classification_output_path, 1)

        # Log the completion of the operation
        self.log_operation("Correct Cell Classification", start_time=start_time)

    def load_images(self, folder_path, image_type="other"):
        """Load images and display them in the scrollable area"""
        # Clear all images in the image layout
        for i in reversed(range(self.image_layout.count())):
            widget_to_remove = self.image_layout.takeAt(i).widget()  # Use takeAt() to remove
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()  # Delete the widget

        # Save the current folder path to ensure the display_image method can use it
        self.current_folder_path = folder_path  # Assign here

        # Load image files
        self.image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else float('inf')
        )

        if not self.image_files:
            return  # If no images, return directly

        # Calculate the maximum value of the scrollbar based on the number of images and the number of images displayed per page
        max_value = max(1, (len(self.image_files) + 3) // 4)  # Display 4 images per page
        self.scroll_bar.setMaximum(max_value)  # Set the scrollbar maximum value to the number of pages
        self.scroll_bar.setEnabled(True)  # Enable the scrollbar

        # Reset the scrollbar to the top (minimum value)
        self.scroll_bar.setValue(1)

        # Load the first set of images
        self.display_images(0)

    def display_images(self, start_index=0):
        """Display 2x2 images starting from start_index, and show the file name below each image"""
        total_images = len(self.image_files)
        num_images = min(max(total_images - start_index, 0), 4)  # Ensure num_images is not less than 0 and does not exceed 4

        if num_images == 0:
            return  # No images, return directly

        # Clear the current layout of images
        for i in reversed(range(self.image_layout.count())):
            widget_to_remove = self.image_layout.takeAt(i).widget()  # Use takeAt() to remove
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()  # Delete the widget

        # Create a canvas for displaying images
        canvas = MplCanvas(self, width=8.00, height=8.00, dpi=100)

        # Use subplots to create a 2x2 layout
        axes = canvas.fig.subplots(2, 2, gridspec_kw={'wspace': 0.2, 'hspace': 0.2})  # Add spacing between rows and columns
        axes = axes.flatten()  # Flatten the 2x2 array into a 1D array

        # Iterate through the subplots and load images
        for idx, ax in enumerate(axes):
            if idx < num_images:
                image_file = self.image_files[start_index + idx]
                image_path = os.path.join(self.current_folder_path, image_file)

                try:
                    # Load the image
                    img_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img_data is not None:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

                        # Display the image
                        ax.imshow(img_data, aspect='auto')  # Maintain image aspect ratio
                        ax.axis('off')  # Hide the axes

                        # Display the file name below the image
                        ax.set_title(image_file, fontsize=10, pad=5, loc='center')
                    else:
                        raise ValueError("Image Load Error")
                except Exception as e:
                    # If the image fails to load, display a warning text
                    ax.text(0.5, 0.5, 'Failed to Load Image', ha='center', va='center',
                        fontsize=12, color='red', transform=ax.transAxes
                    )
                    ax.axis('off')
            else:
                # Hide subplots without assigned images
                ax.axis('off')

        # Adjust the canvas margins to avoid image compression
        canvas.fig.subplots_adjust(wspace=0.1, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        # Add the canvas to the layout
        self.image_layout.addWidget(canvas)
        canvas.draw_idle()  # Refresh the canvas

    def scroll_value_changed(self):
        """Update the displayed images when the scrollbar value changes"""
        value = (self.scroll_bar.value() - 1) * 4  # Display 4 images at a time

        # Protective check to ensure the value is within a reasonable range
        if value < 0:
            value = 0
        elif value >= len(self.image_files):
            value = len(self.image_files) - 4  # Display the last page of images, up to 4

        # Call display_images to display images starting from value
        self.display_images(value)





    def bio_compute_cell_pca(self, cells_info, path):
        scaler = StandardScaler()
        cell_info_scaled = scaler.fit_transform(cells_info)

        pca_all = PCA()
        pca_all.fit(cell_info_scaled)

        explained_variance_ratio = pca_all.explained_variance_ratio_
        explained_variance_df = pd.DataFrame({
            'Principal Component': ['PC' + str(i + 1) for i in range(len(explained_variance_ratio))],
            'Explained Variance Ratio': explained_variance_ratio
        })
        explained_variance_file = os.path.join(path, 'Explained_Variance_Ratio_All_PC.xlsx')
        explained_variance_df.to_excel(explained_variance_file, index=False)

        # Set DPI and make all text bold and larger
        plt.figure(figsize=(10, 6), dpi=300)
        bars = plt.bar(
            explained_variance_df['Principal Component'],
            explained_variance_df['Explained Variance Ratio'] * 100,
            width=0.4, color='lightgreen'
        )
        plt.xlabel('Principal Component', fontsize=18, fontweight='bold')
        plt.ylabel('Explained Variance Ratio (%)', fontsize=18, fontweight='bold')
        plt.title('Explained Variance Ratio for All PC', fontsize=20, fontweight='bold')
        plt.xticks(rotation=0, fontsize=16, fontweight='bold')
        plt.tight_layout()

        for bar, variance in zip(bars, explained_variance_df['Explained Variance Ratio'] * 100):
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{variance:.2f}%', ha='center', color='black', fontsize=14, fontweight='bold'
            )

        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        plot_file_path = os.path.join(path, 'Explained_Variance_Ratio_All_PC.png')
        plt.savefig(plot_file_path, dpi=300)
        plt.close()

        feature_names = {
            'Cell Area': 'S',
            'Maximum Horizontal Length': 'AP',
            'Maximum Vertical Length': 'DV',
            'Horizontal/Vertical Length Ratio': 'AP/DV',
            'Cell Perimeter': 'P',
            'Approximate Polygon Vertex Count': 'N',
            'Fitted Ellipse Minor Axis': 'Sa',
            'Fitted Ellipse Major Axis': 'La',
            'Ellipse Major/Minor Axis Ratio': 'La/Sa',
            'Fitted Ellipse Angle': 'θ',
            'Circumscribed Circle Radius': 'R_c',
            'Inscribed Circle Radius': 'r_i',
            'Center X Coordinate': 'Cx',
            'Center Y Coordinate': 'Cy',
            'Cell Left Boundary': 'Lb',
            'Cell Left Boundary': 'Rb',
            'Circularity': 'C',
            'P-Value': 'P-value'
        }

        pca = PCA(n_components=0.85)
        cell_info_pca = pca.fit_transform(cell_info_scaled)

        explained_variance_selected = pca.explained_variance_ratio_
        components = pca.components_

        for i in range(components.shape[0]):
            contributions = components[i]
            contributions_squared = contributions ** 2
            feature_abbreviations = [feature_names.get(col, col) for col in cells_info.columns]
            component_contributions = pd.DataFrame({
                'Feature': feature_abbreviations,
                'Contribution': contributions,
                'Contribution Squared': contributions_squared
            })

            component_file_path = os.path.join(path, f'PCA_Component_{i + 1}.xlsx')
            component_contributions.to_excel(component_file_path, index=False)

            plt.figure(figsize=(10, 6), dpi=300)
            bars = plt.bar(
                component_contributions['Feature'],
                component_contributions['Contribution Squared'] * 100,
                width=0.4, color='skyblue'
            )
            plt.xlabel('Feature', fontsize=18, fontweight='bold')
            plt.ylabel('Contribution Squared (%)', fontsize=18, fontweight='bold')
            plt.title(f'Contribution Squared for PC {i + 1}', fontsize=20, fontweight='bold')
            plt.xticks(rotation=0, ha='right', fontsize=16, fontweight='bold')
            plt.tight_layout()

            for bar, contribution in zip(bars, component_contributions['Contribution Squared'] * 100):
                plt.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{contribution:.2f}%', ha='center', color='black', fontsize=14, fontweight='bold'
                )

            plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
            plot_file_path = os.path.join(path, f'PCA_Component_{i + 1}_Contribution_Squared.png')
            plt.savefig(plot_file_path, dpi=300)
            plt.close()

        pca_columns = ['PC' + str(i + 1) for i in range(cell_info_pca.shape[1])]
        cell_scores = pd.DataFrame(cell_info_pca, index=cells_info.index, columns=pca_columns)
        cell_scores['PC_score'] = np.sum(cell_scores * pca.explained_variance_ratio_, axis=1)
        cell_scores.to_excel(os.path.join(path, 'Cells_info_PCA.xlsx'), sheet_name='Cell1')

        cells_info_pca = cell_scores.sort_values(axis=0, by='PC_score', ascending=False)

        return cells_info_pca


    def bio_clustering_k_mean(self, cells_info_pca, path):
        # Extract data obtained from PCA
        fac_index = cells_info_pca.index

        # Automatically select all columns starting with 'PC' as principal components
        components = [col for col in cells_info_pca.columns[:-1] if col.startswith('PC')]

        # Extract data for all principal components
        data = cells_info_pca[components].values

        # Use K-Means clustering, set to two clusters
        kmeans = KMeans(n_clusters=2, n_init=10)
        label = kmeans.fit_predict(data)

        # Plot scatter plot of the first two principal components
        plt.figure(figsize=(10, 8), dpi=300)  # Set high resolution
        plt.scatter(data[:, 0], data[:, 1], c=label, cmap='viridis', s=5)
        plt.xlabel('PC1', fontsize=18, fontweight='bold')
        plt.ylabel('PC2', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')

        # Show the right and top border axes
        ax = plt.gca()
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)

        # Set axis range, leave some margins to ensure the right border is displayed
        ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
        ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)

        # Fit a decision boundary using logistic regression
        log_reg = LogisticRegression()
        log_reg.fit(data[:, :2], label)  # Use only the first two principal components

        # Create a grid to plot the 2D decision boundary
        x_values = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
        y_values = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_values, y_values)
        grid = np.c_[xx.ravel(), yy.ravel()]
        probabilities = log_reg.predict_proba(grid)[:, 1].reshape(xx.shape)

        # Plot the decision boundary
        plt.contour(xx, yy, probabilities, levels=[0.5], linestyles=['--'], colors='r')

        # Use tight_layout and increase padding appropriately
        plt.tight_layout(pad=1)

        # Save the image
        plt.savefig(os.path.join(path, 'cells_PCA_2D.png'), dpi=300)
        plt.close()

        # Output clustering results
        if np.sum(label == 1) > np.sum(label == 0):
            small_cells_index = np.where(label == 1)
            big_cells_index = np.where(label == 0)
            label = np.where(label == 0, 1, 0)  # Reverse labels
        else:
            big_cells_index = np.where(label == 1)
            small_cells_index = np.where(label == 0)
            label = np.where(label == 1, 1, 0)  # Keep labels

        # Generate prediction results
        big_cells_predict = list(fac_index[big_cells_index])
        small_cells_predict = list(fac_index[small_cells_index])

        # Generate an Excel file for the classification results
        results = pd.DataFrame({
            'Cell_Index': fac_index,
            'Cluster_Label': label,
            'Cell_Type': ['mes cell' if lbl == 1 else 'epi cell' for lbl in label]
        })
        print(results)

        # Save results to Excel
        results.to_excel(os.path.join(path, 'cells_clustering_results.xlsx'), index=False)

        return big_cells_predict, small_cells_predict

    def show_plot_track_window(self):
        analysis_data_dir = os.path.join(config.quantitative_analysis_output_path,
                                         'all_cell_quantitative_analysis_output')
        if not os.path.exists(analysis_data_dir) or not os.listdir(analysis_data_dir):
            QMessageBox.warning(self, "Notice", "Quantitative analysis data not found. Please generate it first.")
            return
        # Record start time
        start_time = time.time()

        # Log the start of cell classification
        self.log_operation("Display Cell Tracking Trajectories", start_time=None)

        img_folder = config.Img_path
        npy_folder = config.npy_folder_path

        self.plot_track_window = PlotTrackWindow(img_folder, npy_folder)
        self.plot_track_window.show()
        self.log_operation("Display Cell Tracking Trajectories", start_time=start_time)

    def run_cell_tracking(self, cells_info, progress_start=0, progress_end=100):
        """
        Process cell tracking and update the progress bar within the specified range.

        :param cells_info: DataFrame containing cell information to be tracked.
        :param progress_start: Starting percentage of the progress bar update.
        :param progress_end: Ending percentage of the progress bar update.
        :return: DataFrame of merged tracking data.
        """
        # Ensure all indices are in string format
        cells_info.index = cells_info.index.map(str)

        # Get the content of Cells_info
        path = config.cell_track_output_path

        # Initialize start_index
        start_index = 1
        merge_only = False

        # Determine the situation of the 'leading_edge' column in cells_info
        if 'leading_edge' not in cells_info.columns:
            # If there is no 'leading_edge' column
            folder_prefix = "All_cell_"
            output_subfolder = "All_cell_tracking/"
            name = 'all_cell'
        elif (cells_info['leading_edge'] <= 0).all():
            folder_prefix = "mes_cell_"
            output_subfolder = "mes_cell_tracking/"
            name = 'mes_cell'

            # Check if the merged file for epithelial cells exists
            small_cell_file = os.path.join(path, "epi_cell_tracking",
                                           "epi_cell_merged_tracking_results.xlsx")
            if os.path.exists(small_cell_file):
                small_cell_df = pd.read_excel(small_cell_file, header=0)
                if not small_cell_df.empty:
                    start_index = small_cell_df.iloc[:, 0].max() + 1
            # else:
            # No merged file for epithelial cells found, start numbering from 1

        elif (cells_info['leading_edge'] > 0).all():
            folder_prefix = "epi_cell_"
            output_subfolder = "epi_cell_tracking/"
            name = 'epi_cell'

            # Check if the merged file for cells exists
            big_cell_file = os.path.join(path, "mes_cell_tracking", "mes_cell_merged_tracking_results.xlsx")
            if os.path.exists(big_cell_file):
                big_cell_df = pd.read_excel(big_cell_file, header=0)
                if not big_cell_df.empty:
                    start_index = big_cell_df.iloc[:, 0].max() + 1
            # else:
            # No merged file for cells found, start numbering from 1

        output_path = os.path.join(path, output_subfolder)
        os.makedirs(output_path, exist_ok=True)

        # Set pandas display options (optional, affects debug output)
        pd.set_option('display.max_row', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.unicode.east_asian_width', True)
        pd.set_option('display.width', 180)

        # Tracking and matching logic

        # Extract all unique frame numbers
        fig_ids = sorted(set(
            int(re.search(r'Cell(\d+)_', idx).group(1)) for idx in cells_info.index if re.search(r'Cell(\d+)_', idx)))
        total_figs = len(fig_ids)
        if total_figs == 0:
            QMessageBox.warning(self, 'Warning', 'No valid frames found for tracking.')
            return pd.DataFrame()

        for idx, fig_id in enumerate(fig_ids):
            # Calculate current progress and update the progress bar
            if total_figs > 0:
                local_progress = (idx + 1) / total_figs  # 0 to 1
                global_progress = progress_start + local_progress * (progress_end - progress_start)
                progress_value = min(int(global_progress), progress_end)
            else:
                progress_value = progress_end

            self.show_progress_bar(f"Tracking frame {fig_id} for {name}...")
            self.progress_dialog.setValue(progress_value)
            QApplication.processEvents()  # Force refresh of the event loop

            # Get cell information for the current and next frames
            cells_info_fig1 = cells_info[cells_info.index.str.contains(f'Cell{fig_id}_')]
            cells_info_fig2 = cells_info[cells_info.index.str.contains(f'Cell{fig_id + 1}_')]
            cells_info_fig1_index = list(cells_info_fig1.index)
            cells_info_fig2_index = list(cells_info_fig2.index)

            # Calculate relationships between cells
            cells_relation = np.zeros((len(cells_info_fig1_index), len(cells_info_fig2_index)))
            cells_nature = [
                'Cell Area', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio',
                'Cell Perimeter', 'Approximate Polygon Vertex Count', 'Fitted Ellipse Minor Axis',
                'Fitted Ellipse Major Axis', 'Ellipse Major/Minor Axis Ratio', 'Fitted Ellipse Angle',
                'Circumscribed Circle Radius', 'Inscribed Circle Radius', 'Center X Coordinate',
                'Center Y Coordinate', 'Cell Left Boundary', 'Cell Right Boundary', 'Circularity', 'P-Value'
            ]

            cells_weight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]

            max_euclidean_distance = 0
            for cell1_id in range(len(cells_info_fig1_index)):
                cell1_id_index = cells_info_fig1_index[cell1_id]
                cell1_info = list(cells_info_fig1.loc[cell1_id_index, cells_nature])
                for cell2_id in range(len(cells_info_fig2_index)):
                    cell2_id_index = cells_info_fig2_index[cell2_id]
                    cell2_info = list(cells_info_fig2.loc[cell2_id_index, cells_nature])

                    max_distance = max(cell1_info[1], cell1_info[2])

                    if abs(cell2_info[12] - cell1_info[12]) > max_distance or abs(
                            cell2_info[13] - cell1_info[13]) > max_distance:
                        cells_relation[cell1_id, cell2_id] = np.nan
                        continue

                    # Calculate Euclidean distance
                    cell1_cell2_distance = 0
                    for cell_nature_id in range(len(cells_nature)):
                        cell1_cell2_distance += cells_weight[cell_nature_id] * (
                                cell2_info[cell_nature_id] - cell1_info[cell_nature_id]) ** 2
                    cell1_cell2_distance = np.sqrt(cell1_cell2_distance)

                    cells_relation[cell1_id, cell2_id] = cell1_cell2_distance

            # Save cells_relation to an Excel file
            relation_df = pd.DataFrame(cells_relation, index=cells_info_fig1_index, columns=cells_info_fig2_index)
            relation_file_path = os.path.join(output_path, f'{folder_prefix}Cell_Relation_{fig_id}_{fig_id + 1}.xlsx')
            relation_df.to_excel(relation_file_path)

            df = pd.DataFrame(cells_relation, index=cells_info_fig1_index, columns=cells_info_fig2_index)

            max_euclidean_distance = cells_info['Circumscribed Circle Radius'].max()

            # Use the Hungarian algorithm for matching
            df_filled = df.fillna(max_euclidean_distance)
            row_ind, col_ind = linear_sum_assignment(df_filled.values)

            matching_result = pd.DataFrame({
                f'Frame{fig_id}': [df.index[r] if r < len(df.index) else f"Cell{fig_id}_" for r in row_ind],
                f'Frame{fig_id + 1}': [df.columns[c] if c < len(df.columns) else f"Cell{fig_id + 1}_" for c in col_ind],
                'Difference': [df_filled.iat[r, c] for r, c in zip(row_ind, col_ind)]
            })

            # Number the matching results starting from start_index
            matching_result.insert(0, 'Index', range(start_index, start_index + len(matching_result)))

            # Filter out rows with a difference value equal to max_euclidean_distance
            matching_result = matching_result[matching_result['Difference'] != max_euclidean_distance]

            # Save the matching results
            filtered_output_file_path = os.path.join(output_path,
                                                     f'{folder_prefix}Matching_Results_new_{fig_id}_{fig_id + 1}.xlsx')
            matching_result.to_excel(filtered_output_file_path, index=False)

            # Update start_index for the next numbering
            start_index += len(matching_result)

        # Merge tracking results
        matching_files = [f for f in os.listdir(output_path) if
                          re.match(rf'{folder_prefix}Matching_Results_new_\d+_\d+\.xlsx', f)]
        matching_frames = sorted(
            set(int(re.search(r'\d+', f).group(0)) for f in matching_files if re.search(r'\d+', f)))
        start_frame = min(matching_frames) if matching_frames else 1
        files_count = len(matching_frames)

        # Initialize and load the first file
        if matching_files:
            first_file_path = os.path.join(output_path,
                                           f'{folder_prefix}Matching_Results_new_{start_frame}_{start_frame + 1}.xlsx')
            data_final = self.load_and_process_data(first_file_path)
            data_final.columns = ['Index', f'Frame{start_frame}', f'Frame{start_frame + 1}']

            # Construct new column names to be added
            all_new_cols = [f'Frame{i + 1}' for i in range(start_frame + 1, start_frame + files_count - 1)]

            # Use pd.concat to add columns at once to avoid fragmentation
            extra_columns_df = pd.DataFrame({col: [None] * len(data_final) for col in all_new_cols})
            data_final = pd.concat([data_final, extra_columns_df], axis=1)

            # Process subsequent files
            for i in range(start_frame + 1, start_frame + files_count - 1):
                next_file_path = os.path.join(output_path, f'{folder_prefix}Matching_Results_new_{i}_{i + 1}.xlsx')
                data_next = self.load_and_process_data(next_file_path)
                data_next.columns = ['Index', f'Frame{i}', f'Frame{i + 1}']

                for index, row in data_next.iterrows():
                    match_index = data_final[data_final[f'Frame{i}'] == row[f'Frame{i}']].index
                    if not match_index.empty:
                        data_final.at[match_index[0], f'Frame{i + 1}'] = row[f'Frame{i + 1}']

        else:
            data_final = pd.DataFrame()

        # Save the merged results
        merged_output_path = os.path.join(output_path, f'{folder_prefix}merged_tracking_results.xlsx')
        data_final.to_excel(merged_output_path, index=False)

        # Return the merged data
        return data_final
    
    def load_and_process_data(self, file_path):
        # Load the Excel file
        data = pd.read_excel(file_path)
        # Remove the unnecessary "Difference" column
        data.drop(columns=['Difference'], inplace=True)
        return data

    def generate_and_notify(self):

        # Record the start time
        start_time = time.time()

        # Create and display the initial progress bar
        self.show_progress_bar("Generating quantitative analysis data, please wait...")
        # Set the initial value of the progress bar
        self.progress_dialog.setValue(1)

        tracking_data_path = os.path.join(config.cell_track_output_path, 'all_cell_tracking', 'all_cell_merged_tracking_results.xlsx')
        if not os.path.exists(tracking_data_path):
            self.progress_dialog.close()  # ✅ Add this line to close the progress bar
            QMessageBox.warning(self, "Notice", "Cell tracking results not found. Please perform cell tracking first.")
            return

        # Log the start of generating quantitative analysis data
        self.log_operation("Generating quantitative analysis data", start_time=None)

        all_cells_output_dir = self.generate_quantitative_analysis_data('all_cell_tracking',
                'all_cell_merged_tracking_results_updated.xlsx',
            'all_cell_merged_tracking_results.xlsx', 'all_cell_quantitative_analysis_output'
        )
        self.progress_dialog.setValue(100)
        self.progress_dialog.close()

        # Notify the user that all data has been generated
        QMessageBox.information(
            self, "Data Generation Complete",
            f"Quantitative analysis data has been successfully generated and saved in the following directory:\n"
            f"All Cells:{all_cells_output_dir}\n"
        )
        self.log_operation("Generating quantitative analysis data", start_time=start_time)


    def generate_quantitative_analysis_data(self, tracking_folder, updated_filename, default_filename,
                                            output_folder_name):
        # Check and select the Excel file path to load
        updated_file1_path = os.path.join(config.cell_track_output_path, tracking_folder, updated_filename)
        default_file1_path = os.path.join(config.cell_track_output_path, tracking_folder, default_filename)

        # Use the updated file if it exists; otherwise, use the default file
        if os.path.exists(updated_file1_path):
            file1_path = updated_file1_path
        else:
            file1_path = default_file1_path

        # Load file2: Path to Cells_info.xlsx
        file2_path = os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx')

        # Check if Cells_info.xlsx exists
        if not os.path.exists(file2_path):
            raise FileNotFoundError(f"Cells_info.xlsx does not exist in the specified path: {file2_path}.")

        # Read the Excel files, load all sheets
        file1_data = pd.read_excel(file1_path, sheet_name=None)  # Read all sheets
        file2_data = pd.read_excel(file2_path, sheet_name=None)  # Read all sheets

        # Print the sheet names in file2_data
        #print(f"Sheet names in Cells_info.xlsx: {file2_data.keys()}")

        # Get the sheets
        file1_sheet1 = file1_data['Sheet1']  # Read Sheet1 from file1
        file2_sheet1 = file2_data['Cell']  # Read Sheet1 from file2

        # Print all column names in file2_sheet1
        #print("Column names in file2_sheet1: ", file2_sheet1.columns)

        # Check if the column 'Cell Index' exists
        if 'Cell Index' not in file2_sheet1.columns:
            raise KeyError(f"'The column 'Cell Index' is missing in 'Cells_info.xlsx'. Available columns are:{file2_sheet1.columns}")

        # Create the output directory (if it doesn't exist) and clear its contents
        output_dir = os.path.join(config.quantitative_analysis_output_path, output_folder_name)

        # If the directory exists, delete all files and subfolders in it
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Delete files or symbolic links
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete subfolders
                except Exception as e:
                    print(f"Error occurred while deleting file '{file_path}': {e}")

        # Recreate the output directory
        os.makedirs(output_dir, exist_ok=True)
        # Get the total number of tracks
        total_tracks = len(file1_sheet1)

        # Iterate through each row in the tracking data
        for index, row in file1_sheet1.iterrows():
            # Extract the cell IDs in the current row
            cell_ids = row[1:].values  # Extract all cell IDs starting from the second column

            # Get the corresponding sequence number
            sequence_number = row['Index']

            # Create a DataFrame to store the matching rows for the current track
            track_df = pd.DataFrame()

            # Find the corresponding rows in the cell information data
            for i, cell_id in enumerate(cell_ids):
                matched_row = file2_sheet1[file2_sheet1['Cell Index'] == cell_id]
                if not matched_row.empty:
                    matched_row = matched_row.copy()
                    # Add the frame number (1, 2, 3, ...)
                    matched_row.insert(0, 'Index', i + 1)  # Insert 'Index' as the first column
                    # Append the matched row to the track DataFrame
                    track_df = pd.concat([track_df, matched_row], ignore_index=True)

            # Save the results to a new Excel file
            output_filename = f'track_{sequence_number}_quantitative_analysis.xlsx'
            if not hasattr(self, 'cells_info') or self.cells_info.empty:
                QMessageBox.warning(self, "Notice", "Please extract cell features before performing this operation.")
                return
            output_path = os.path.join(output_dir, output_filename)
            track_df.to_excel(output_path, index=False)

            # Create and display progress bar text, updating the progress bar for each cell's sequence number
            self.show_progress_bar(f"Quantitative analysis data for cell {sequence_number} has been generated...")

            # Update the progress bar
            progress_value = int((index + 1) / total_tracks * 100)
            self.progress_dialog.setValue(progress_value)

        # Return the directory where the files are saved
        return output_dir


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
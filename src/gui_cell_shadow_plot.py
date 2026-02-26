import sys
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QLineEdit, QPushButton,
    QMessageBox, QHBoxLayout, QScrollArea, QGridLayout, QComboBox, QSizePolicy, QSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
from PyQt5.QtCore import Qt
from matplotlib import colormaps
import numpy as np
from datetime import datetime  # Add import for datetime
import config  # Ensure you have a config.py file with required paths defined
from scipy.stats import linregress

# Auto-select font path based on operating system
if sys.platform.startswith('win'):
    font_path = "C:\\Windows\\Fonts\\simhei.ttf"  # Windows system
elif sys.platform.startswith('darwin'):
    font_path = "/System/Library/Fonts/PingFang.ttc"  # macOS system
else:
    font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"  # Linux

# Check if font file exists
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.family'] = 'sans-serif'

# Resolve minus sign display issue
plt.rcParams['axes.unicode_minus'] = False


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        # Mouse event variables
        self.drag_start = None  # Record the starting point of drag

        # Save initial range
        self.initial_xlim = (-10, 10)
        self.initial_ylim = (-10, 10)
        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)

        # Hide coordinate axes
        self.ax.axis('off')  # Add this line

        # Bind mouse events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        if event.inaxes:  # Mouse click within canvas
            self.drag_start = (event.xdata, event.ydata)  # Record starting point

    def on_motion(self, event):
        if self.drag_start is None or not event.inaxes:
            return
        x_start, y_start = self.drag_start
        dx = x_start - event.xdata
        dy = y_start - event.ydata

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        self.ax.set_xlim([x_min + dx, x_max + dx])
        self.ax.set_ylim([y_min + dy, y_max + dy])
        self.draw()

    def on_release(self, event):
        self.drag_start = None

    def on_scroll(self, event):
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        scale_factor = 0.9 if event.button == 'up' else 1.1
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        x_range = (x_max - x_min) * scale_factor
        y_range = (y_max - y_min) * scale_factor

        self.ax.set_xlim([x_mid - x_range / 2, x_mid + x_range / 2])
        self.ax.set_ylim([y_mid - y_range / 2, y_mid + y_range / 2])
        self.draw()

    def reset_view(self):
        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)
        self.draw()


class CellShadowLineGUI(QMainWindow):
    def __init__(self, quantitative_analysis_output_path, all_cell_quantitative_analysis_output, cell_track_output_pictures):
        super().__init__()
        self.setWindowTitle("Multi-Cell Quantitative Analysis")
        self.setGeometry(100, 100, 1300, 600)
        self.category_inputs = []  # Store dynamically added category input boxes
        self.base_path = all_cell_quantitative_analysis_output
        self.image_folder = cell_track_output_pictures
        self.quantitative_analysis_output_path = quantitative_analysis_output_path

        # Available features
        self.features = [
            'Cell Area', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio',
            'Cell Perimeter', 'Approximate Polygon Vertex Count',
            'Fitted Ellipse Minor Axis', 'Fitted Ellipse Major Axis', 'Ellipse Major/Minor Axis Ratio',
            'Fitted Ellipse Angle', 'Circumscribed Circle Radius',
            'Inscribed Circle Radius', 'Center X Coordinate', 'Center Y Coordinate', 'Cell Left Boundary',
            'Cell Right Boundary', 'Circularity', 'P-Value', 'leading_edge', 'γ'
        ]

        # Main layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.bottom_layout = QHBoxLayout()

        # Left section
        self.left_layout = QVBoxLayout()

        # Feature selection, time interval input and add category button layout
        feature_layout = QHBoxLayout()
        self.feature_label = QLabel("Select Feature:")
        self.feature_combo = QComboBox(self)
        self.feature_combo.setFixedWidth(100)  # Set dropdown width to 100 pixels
        self.feature_combo.addItems(self.features)

        self.time_interval_label = QLabel("Time Interval (s):")
        self.time_interval_spinbox = QSpinBox(self)
        self.time_interval_spinbox.setRange(1, 100000)  # Set range
        self.time_interval_spinbox.setValue(60)  # Default value is 60 seconds
        self.time_interval_spinbox.setFixedWidth(50)

        self.add_category_input_button = QPushButton("Add Group", self)
        self.add_category_input_button.setFixedWidth(100)  # Set button width to 100 pixels

        self.add_category_input_button.clicked.connect(self.add_category_input)

        # Add feature selection, time interval input and button to horizontal layout
        feature_layout.addWidget(self.feature_label)
        feature_layout.addWidget(self.feature_combo)
        feature_layout.addWidget(self.time_interval_label)
        feature_layout.addWidget(self.time_interval_spinbox)
        feature_layout.addWidget(self.add_category_input_button)

        self.left_layout.addLayout(feature_layout)

        # Input section (scrollable area supporting dynamic addition)
        scroll_area = QScrollArea(self)
        self.form_widget = QWidget()

        # Use QGridLayout as layout
        self.form_layout = QGridLayout(self.form_widget)
        self.form_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Ensure arrangement starts from top-left

        # Set horizontal and vertical spacing to 0
        self.form_layout.setHorizontalSpacing(5)
        self.form_layout.setVerticalSpacing(5)

        # Remove layout content margins
        self.form_layout.setContentsMargins(5, 5, 5, 5)

        self.form_widget.setLayout(self.form_layout)
        scroll_area.setWidget(self.form_widget)
        scroll_area.setWidgetResizable(True)
        self.left_layout.addWidget(scroll_area)

        # Default add category 1
        self.add_category_input(is_default=True)

        # Plot button
        self.plot_button = QPushButton("Plot", self)
        self.plot_button.clicked.connect(self.plot_selected_feature)
        self.left_layout.addWidget(self.plot_button)

        # Display the image with minimum numbering
        self.min_image_canvas = MplCanvas(self, width=4, height=4, dpi=100)
        self.min_image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_layout.addWidget(self.min_image_canvas)

        # Add left layout
        self.bottom_layout.addLayout(self.left_layout, 5)

        # Right section (display chart)
        self.right_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.right_layout.addWidget(self.canvas)
        self.bottom_layout.addLayout(self.right_layout, 8)

        # Add left and right layouts to main layout
        main_layout.addLayout(self.bottom_layout)

        # Display image
        self.display_min_image()

    def add_category_input(self, is_default=False):
        """Dynamically add input boxes, arranged from top-left"""
        # Create input box
        input_line = QLineEdit(self)
        input_line.setPlaceholderText("Enter Cell ID (comma-separated):")
        input_line.setFixedWidth(155)  # Input box width fixed at 155 pixels

        self.category_inputs.append(input_line)

        # Dynamically calculate row and column
        row = (len(self.category_inputs) - 1) // 2  # Display two categories per row
        col = (len(self.category_inputs) - 1) % 2  # Current column: 0 or 1

        # Default category numbering
        category_number = 1 if is_default else len(self.category_inputs)

        # Create category label
        category_label = QLabel(f"Group {category_number}:")
        category_label.setFixedWidth(60)  # Set label fixed width

        # Remove label and input box margins and padding
        category_label.setStyleSheet("margin: 0px; padding: 0px;")
        input_line.setStyleSheet("margin: 0px; padding: 0px;")

        # Add controls to grid layout
        self.form_layout.addWidget(category_label, row, col * 2, alignment=Qt.AlignLeft)  # Label in left column
        self.form_layout.addWidget(input_line, row, col * 2 + 1, alignment=Qt.AlignLeft)  # Input box in right column

    def load_track_data(self, track_numbers, feature):
        """Load Excel data based on track numbers and return DataFrame for specified feature"""
        base_path = self.base_path
        data_frames = []
        for track_number in track_numbers:
            file_name = f"track_{track_number}_quantitative_analysis.xlsx"
            file_path = os.path.join(base_path, file_name)
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                if feature in df.columns:
                    data_frames.append(df[feature])
                else:
                    QMessageBox.warning(self, "Invalid Feature",
                                        f"The feature '{feature}' does not exist in the trajectory file '{file_name}'.")
                    return None
            else:
                QMessageBox.warning(self, "File Not Found", f"The trajectory file {file_name} does not exist.")
                return None
        if not data_frames:
            return None
        return pd.concat(data_frames, axis=1)

    def calculate_mean_std(self, data_frame):
        """Calculate mean and standard deviation for each time point"""
        mean_values = data_frame.mean(axis=1)
        std_values = data_frame.std(axis=1)
        return mean_values, std_values

    def generate_soft_colors(self, num_colors):
        if num_colors <= 1:
            return [(0.121, 0.466, 0.705, 1)]  # matplotlib default blue

        # Use Set2 colormap, generate colors with higher contrast
        cmap = colormaps['Set2']  # Get Set2 colormap
        if num_colors > 8:
            # If more than 8 colors needed, use a larger palette
            cmap = colormaps['tab20']  # Use tab20 palette with more colors
        colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
        return colors

    def plot_selected_feature(self):
        """Determine which plotting function to call based on selected feature."""
        selected_feature = self.feature_combo.currentText()
        self.canvas.ax.clear()  # Clear previous plot

        if selected_feature == 'γ':
            self.plot_msd_boxplot()
        else:
            self.plot_lines()

    def plot_lines(self):
        """Plot optimized line chart with rectangular boundaries"""
        self.canvas.ax.clear()

        # Get user-selected feature
        selected_feature = self.feature_combo.currentText()

        # Dynamically generate colors
        num_categories = len(self.category_inputs)
        colors = self.generate_soft_colors(num_categories)

        all_mean_values = []  # Collect all mean values to determine unified range
        all_time_points = []

        for idx, input_line in enumerate(self.category_inputs):
            track_numbers = [int(x.strip()) for x in input_line.text().split(',') if x.strip().isdigit()]
            if not track_numbers:
                continue

            # Load data
            data_frame = self.load_track_data(track_numbers, selected_feature)
            if data_frame is None:
                continue

            # Calculate mean and standard deviation
            mean_values, std_values = self.calculate_mean_std(data_frame)

            # Collect time points and mean values
            time_points = range(1, len(mean_values) + 1)
            all_mean_values.extend(mean_values)
            all_time_points.extend(time_points)

            # Plot line chart with shadow
            color = colors[idx]  # Take color from generated color list
            self.canvas.ax.plot(time_points, mean_values, label=f"Group {idx + 1}", color=color, linewidth=2.5)
            self.canvas.ax.fill_between(time_points, mean_values - std_values, mean_values + std_values,
                                        color=color, alpha=0.3)

        if not all_mean_values or not all_time_points:
            QMessageBox.warning(self, "No Data", "No data available to plot.")
            return

        # Ensure origin is included
        x_min, x_max = min(all_time_points, default=0), max(all_time_points, default=1)
        y_min, y_max = min(all_mean_values, default=0), max(all_mean_values, default=1)

        # Dynamically increase boundary margins
        x_margin = 0.05 * (x_max - x_min) if x_max != x_min else 1
        y_margin = 0.05 * (y_max - y_min) if y_max != y_min else 1

        # Set axis range
        self.canvas.ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self.canvas.ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Set complete rectangular border
        for spine in ['top', 'bottom', 'left', 'right']:
            self.canvas.ax.spines[spine].set_visible(True)
            self.canvas.ax.spines[spine].set_color('#4D4D4D')  # Border color
            self.canvas.ax.spines[spine].set_linewidth(1.2)  # Border line width

        # Set axis position to default to avoid axis moving inside data
        self.canvas.ax.spines['left'].set_position(('outward', 0))
        self.canvas.ax.spines['bottom'].set_position(('outward', 0))
        self.canvas.ax.spines['right'].set_position(('outward', 0))
        self.canvas.ax.spines['top'].set_position(('outward', 0))

        # Set x and y axis ticks to show only at bottom and left, hide at top and right
        self.canvas.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, direction='in')
        self.canvas.ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, direction='in')

        # Set chart information
        self.canvas.ax.set_title(f"{selected_feature}", fontsize=16, pad=20, color='#2E2E2E')
        self.canvas.ax.set_xlabel("Frame", fontsize=14, labelpad=12, color='#4D4D4D')

        # Adjust legend
        self.canvas.ax.legend(fontsize=12, loc='upper right', frameon=False)

        # Beautify ticks
        self.canvas.ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, color='#4D4D4D')

        # Draw image
        self.canvas.draw()

        # Save image to specified path
        save_dir = self.quantitative_analysis_output_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format time, e.g., 20250110_123456
        save_filename = f"mean_with_variance_{current_time}.pdf"
        save_path = os.path.join(save_dir, save_filename)

        try:
            self.canvas.fig.savefig(save_path, dpi=300)
            QMessageBox.information(self, "Save Successful", f"The line chart has been saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"An error occurred while saving the line chart:\n{e}")

    def plot_msd_boxplot(self):
        """Calculate gamma values for each group and plot boxplot."""
        self.canvas.ax.clear()

        group_gamma_values = []  # Store gamma values for each group
        group_labels = []        # Labels for each group

        # Get global time interval
        time_interval = self.time_interval_spinbox.value()

        if time_interval <= 0:
            QMessageBox.warning(self, "Invalid Input", "Time interval must be a positive integer.")
            return

        # Iterate through each group's input
        for idx, input_line in enumerate(self.category_inputs):
            track_numbers = [int(x.strip()) for x in input_line.text().split(',') if x.strip().isdigit()]
            if not track_numbers:
                continue

            gamma_values = []  # Gamma values for current group

            for track_number in track_numbers:
                file_name = f"track_{track_number}_quantitative_analysis.xlsx"
                file_path = os.path.join(self.base_path, file_name)
                if not os.path.exists(file_path):
                    QMessageBox.warning(self, "File Not Found", f"The trajectory file {file_name} does not exist.")
                    continue

                df = pd.read_excel(file_path)
                if 'Center X Coordinate' not in df.columns or 'Center Y Coordinate' not in df.columns:
                    QMessageBox.warning(self, "Invalid Data",
                                        f"The file {file_name} does not contain required coordinates.")
                    continue

                x_coords = df['Center X Coordinate'].values
                y_coords = df['Center Y Coordinate'].values
                msd_values = []
                total_frames = len(x_coords)

                # Calculate MSD for different time lags
                for lag in range(1, total_frames):
                    displacements = (x_coords[lag:] - x_coords[:-lag]) ** 2 + (y_coords[lag:] - y_coords[:-lag]) ** 2
                    msd = displacements.mean()
                    msd = msd if msd > 0 else 1e-10  # Avoid log(0)
                    msd_values.append(msd)

                if len(msd_values) == 0:
                    QMessageBox.warning(self, "Invalid Data",
                                        f"Track {track_number} has insufficient data for MSD calculation.")
                    continue

                time_lags = np.array(range(1, total_frames)) * time_interval

                log_time = np.log(time_lags)
                log_msd = np.log(msd_values)

                if len(log_time) <= 1:
                    QMessageBox.warning(self, "Invalid Data",
                                        f"Track {track_number} has insufficient data for MSD calculation.")
                    continue

                # Linear fit
                slope, intercept, r_value, _, _ = linregress(log_time, log_msd)
                gamma_values.append(slope)

            if gamma_values:
                group_gamma_values.append(gamma_values)
                group_labels.append(f"Group {idx + 1}")

        if not group_gamma_values:
            QMessageBox.warning(self, "No Data", "No gamma values were calculated.")
            return

        # Plot boxplot
        box = self.canvas.ax.boxplot(
            group_gamma_values, labels=group_labels, patch_artist=True,
            boxprops=dict(facecolor='#4D4D4D', color='#4D4D4D'),
            medianprops=dict(color='yellow'),
            whiskerprops=dict(color='#4D4D4D'),
            capprops=dict(color='#4D4D4D'),
            flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none')
        )

        # Set boxplot colors
        for patch in box['boxes']:
            patch.set_facecolor('#4D4D4D')

        #self.canvas.ax.set_title("γ per Group", fontsize=16, pad=20, color='#2E2E2E')
        self.canvas.ax.set_xlabel("Groups", fontsize=14, labelpad=12, color='#4D4D4D')
        self.canvas.ax.set_ylabel("γ", fontsize=14, labelpad=12, color='#4D4D4D')
        self.canvas.ax.grid(True, linestyle=':', color='grey', alpha=0.7)

        self.canvas.draw()

        # Save boxplot
        save_dir = self.quantitative_analysis_output_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_filename = f"gamma_boxplot_{current_time}.pdf"
        save_path = os.path.join(save_dir, save_filename)

        try:
            self.canvas.fig.savefig(save_path, dpi=300)
            QMessageBox.information(self, "Save Successful", f"The boxplot has been saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"An error occurred while saving the boxplot:\n{e}")

    def display_min_image(self):
        """Find and display the image with minimum numbering in cell_track_output_pictures folder, with mouse interaction support"""
        image_folder = self.image_folder  # Use the passed image_folder
        if not os.path.exists(image_folder):
            QMessageBox.warning(self, 'Error', f'The directory {image_folder} does not exist.')
            return

        # Get all image files
        images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        if not images:
            QMessageBox.warning(self, 'Error', 'No images were found.')
            return

        # Extract numbers from image filenames, find file with minimum number
        def extract_number(image_name):
            match = re.search(r'\d+', image_name)
            return int(match.group()) if match else float('inf')

        min_image = min(images, key=extract_number)
        min_image_path = os.path.join(image_folder, min_image)

        # Load and display image
        self.min_image_canvas.ax.clear()  # Clear current plot
        img_data = plt.imread(min_image_path)
        self.min_image_canvas.ax.imshow(img_data, aspect='auto')  # Display image
        self.min_image_canvas.ax.axis('off')  # Hide coordinate axes

        # Ensure image fills area as much as possible
        self.min_image_canvas.fig.tight_layout(pad=0)  # Reduce whitespace around image
        self.min_image_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Ensure the following variables are defined in your config.py file:
    # quantitative_analysis_output_path
    # all_cell_quantitative_analysis_output
    # cell_track_output_pictures

    quantitative_analysis_output_path = config.quantitative_analysis_output_path
    all_cell_quantitative_analysis_output = config.all_cell_quantitative_analysis_output
    cell_track_output_pictures = config.cell_track_output_pictures

    main_window = CellShadowLineGUI(
        quantitative_analysis_output_path=quantitative_analysis_output_path,
        all_cell_quantitative_analysis_output=all_cell_quantitative_analysis_output,
        cell_track_output_pictures=cell_track_output_pictures
    )
    main_window.show()
    sys.exit(app.exec_())
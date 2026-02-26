import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QLineEdit, QPushButton, \
    QMessageBox, QComboBox, QHBoxLayout, QSizePolicy, QInputDialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import config
import matplotlib
import matplotlib.font_manager as fm
import itertools
from matplotlib.ticker import MaxNLocator
import re
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
import numpy as np
from scipy.stats import linregress
from datetime import datetime









# MplCanvas class with mouse interaction for panning and zooming
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        # mouse interaction variables
        self.drag_start = None  # record drag start position

        # Save initial range
        self.initial_xlim = (-10, 10)
        self.initial_ylim = (-10, 10)
        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)

        # binding mouse events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        if event.inaxes:  # Mouse click inside the canvas
            self.drag_start = (event.xdata, event.ydata)  # Record the starting point

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


class QuantitativeAnalysisGUI(QMainWindow):
    def __init__(self, npy_file_count):
        super().__init__()
        self.setWindowTitle("Quantitative Analysis")
        self.setGeometry(100, 100, 1300, 600)  # Set window size to 1300 width, 600 height

        self.npy_file_count = npy_file_count

        # Create central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create horizontal layout for bottom section
        self.bottom_layout = QHBoxLayout()

        # Left section: Display input box, feature selection, plot button + smallest number image + note
        self.left_layout = QVBoxLayout()

        # Input layout section
        input_layout = QHBoxLayout()
        self.label_track = QLabel("Enter Cell ID (comma-separated):")
        input_layout.addWidget(self.label_track)
        self.track_input = QLineEdit(self)
        input_layout.addWidget(self.track_input)

        # Feature selection dropdown
        self.feature_label = QLabel("Select Features:")
        input_layout.addWidget(self.feature_label)
        self.combo_box = QComboBox(self)
        self.combo_box.addItems([
            'Cell Area', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio', 'Cell Perimeter', 'Approximate Polygon Vertex Count',
            'Fitted Ellipse Minor Axis', 'Fitted Ellipse Major Axis', 'Ellipse Major/Minor Axis Ratio', 'Fitted Ellipse Angle', 'Circumscribed Circle Radius',
            'Inscribed Circle Radius', 'Center X Coordinate', 'Center Y Coordinate', 'Cell Left Boundary', 'Cell Right Boundary', 'Circularity', 'P-Value', 'leading_edge','MSD'
        ])
        self.combo_box.setFixedWidth(100)  # Set fixed width
        input_layout.addWidget(self.combo_box)

        # Add input box and feature selection as a row to the left layout
        self.left_layout.addLayout(input_layout)

        # Create plot button
        self.button = QPushButton("Plot", self)
        self.button.setFixedWidth(500)
        self.button.clicked.connect(self.plot_tracks)

        # Place the plot button on a separate row
        self.left_layout.addWidget(self.button)

        # Display the image with the smallest number
        self.min_image_canvas = MplCanvas(self, width=4, height=4, dpi=100)

        # Set the size policy of the canvas to make it fill the available space as much as possible
        self.min_image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Center the image and ensure it occupies more space
        self.left_layout.addWidget(self.min_image_canvas)

        # Create note label
        self.note_label = QLabel("Note: Cell IDs in the yellow box indicate the cell number.")
        self.note_label.setAlignment(Qt.AlignCenter)  # Set text alignment to center

        # Add note label to the left layout
        self.left_layout.addWidget(self.note_label)

        # Add note label to the left layout
        self.bottom_layout.addLayout(self.left_layout, 5)  # Left: 5 parts, 500 px

        # Right section: Display charts
        self.right_widget = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_widget)
        self.canvas = FigureCanvas(plt.Figure())
        self.right_layout.addWidget(self.canvas)
        self.bottom_layout.addWidget(self.right_widget, 8)

        # Add left and right layouts to the main layout
        main_layout.addLayout(self.bottom_layout)

        self.display_min_image()  # Display the image with the smallest number

    def display_min_image(self):
        """Find and display the image with the smallest number in the cell_track_output_pictures folder, supporting mouse interaction"""
        image_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
        if not os.path.exists(image_folder):
            QMessageBox.warning(self, 'Error', f'The directory {image_folder} does not exist.')
            return

        # Get all image files
        images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.img'))]
        if not images:
            QMessageBox.warning(self, 'Error', 'No images were found.')
            return

        # Extract numbers from image filenames and find the file with the smallest number
        def extract_number(image_name):
            match = re.search(r'\d+', image_name)
            return int(match.group()) if match else float('inf')

        min_image = min(images, key=extract_number)
        min_image_path = os.path.join(image_folder, min_image)

        # Load and display the image
        self.min_image_canvas.ax.clear()  # Clear current drawing
        img_data = plt.imread(min_image_path)
        self.min_image_canvas.ax.imshow(img_data, aspect='auto')
        self.min_image_canvas.ax.axis('off')

        # Ensure the image fills the area as much as possible
        self.min_image_canvas.fig.tight_layout(pad=0)  # Reduce whitespace around the image
        self.min_image_canvas.draw()

    def plot_msd_log_log(self):
        """Plot log-log MSD for multiple tracks with linear fitting and save γ values to an Excel file"""
        if not hasattr(self, 'all_values_dict') or not self.all_values_dict:
            QMessageBox.warning(self, "Invalid Operation",
                                "Please select cells and plot the data before performing MSD calculation.")
            return

        # Pop up input box to get time interval
        time_interval, ok = QInputDialog.getInt(self, "Enter Time Interval",
                                                "Please enter the time interval (in seconds):", 60, 1, 10000, 1)
        if not ok:
            return  # User cancelled the input

        # Create canvas and axis
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        # Use tab20 color palette (20 colors) and enough marker styles
        colors = plt.cm.tab20.colors
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+', 'x', 'p', 'H', '8', '<', '>', '|', '_', '.', ',']
        epsilon = 1e-10  # Small constant to avoid log(0) issues

        gamma_values = []  # Store γ values for each track

        # Iterate over all tracks and plot
        for idx, (track_number, positions) in enumerate(self.all_values_dict.items()):
            # Extract x and y coordinates
            x_coords = positions['Center X Coordinate'].values
            y_coords = positions['Center Y Coordinate'].values
            msd_values = []
            total_frames = len(x_coords)

            # Iterate over different time lags to calculate MSD
            for lag in range(1, total_frames):
                displacements = (x_coords[lag:] - x_coords[:-lag]) ** 2 + (y_coords[lag:] - y_coords[:-lag]) ** 2
                msd = displacements.mean()
                msd_values.append(msd if msd > 0 else epsilon)

            # Log-transform MSD and time intervals
            log_time_lags = np.log(np.array(range(1, total_frames)) * time_interval)
            log_msd = np.log(msd_values)

            # Check if data is sufficient for fitting
            if len(log_time_lags) <= 1 or len(log_msd) <= 1:
                QMessageBox.warning(self, "Invalid Data",
                                    f"Track {track_number} has insufficient data for MSD calculation.")
                continue

            # Linear fitting
            slope, intercept, r_value, _, _ = linregress(log_time_lags, log_msd)
            gamma_values.append((track_number, slope))

            # Determine color and marker to avoid repetition
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # Plot data points and set legend labels
            ax.plot(log_time_lags, log_msd, marker=marker, linestyle='None', color=color, markersize=3,
                    label=f'Cell{track_number}, γ = {slope:.2f}')

            # Plot fitting line without adding to legend
            ax.plot(log_time_lags, intercept + slope * log_time_lags, linestyle='--', color=color, linewidth=1.5)


        # Set title and axis labels with consistent font
        ax.set_title(f"Ln(MSD)", fontsize=16, pad=20, color='#2E2E2E')
        ax.set_xlabel("Ln(t)", fontsize=14, labelpad=12, color='#4D4D4D')

        # Set legend with multiple columns to save space
        ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='black', ncol=2, framealpha=0.9)
        ax.grid(True, linestyle=':', color='grey', alpha=0.7)

        self.canvas.draw()

        # Check if gamma_values is empty
        if not gamma_values:
            QMessageBox.warning(self, "No Data", "No gamma values were calculated. Please check your data.")
            return

        # Generate a unique filename based on timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format current time
        output_dir = config.quantitative_analysis_output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file_name = f'quantitative_analysis_gamma_values_{timestamp}.xlsx'
        output_path = os.path.join(output_dir, output_file_name)

        # Convert gamma_values to DataFrame
        gamma_df = pd.DataFrame(gamma_values, columns=['Track Number', 'Gamma Value'])
        print(gamma_df)  # Check the generated DataFrame

        # Save Excel file
        try:
            gamma_df.to_excel(output_path, index=False)
            QMessageBox.information(self, "Save Successful", f"Gamma values have been saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"An error occurred while saving gamma values:\n{str(e)}")
            print(f"Error details: {e}")

    def plot_tracks(self):
        # Get user input track numbers
        track_numbers = self.track_input.text().strip().split(',')
        track_numbers = [num.strip() for num in track_numbers if num.strip().isdigit()]

        # Iterate over track numbers to extract center point coordinate data
        folders_to_check = [
            'all_cell_quantitative_analysis_output'
        ]

        if not track_numbers:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid cell numbers.")
            return

        selected_column = self.combo_box.currentText()

        # If the user selects "MSD", calculate and plot the MSD graph
        if selected_column == 'MSD':
            # Used to store the center point coordinates of the tracks
            self.all_values_dict = {}

            for track_number in track_numbers:
                track_number = int(track_number)
                file_found = False

                for folder in folders_to_check:
                    file_name = f"track_{track_number}_quantitative_analysis.xlsx"
                    file_path = os.path.join(config.quantitative_analysis_output_path, folder, file_name)

                    if os.path.exists(file_path):
                        # Read Excel file
                        df = pd.read_excel(file_path)

                        # Check if center point coordinate columns exist
                        if 'Center X Coordinate' in df.columns and 'Center Y Coordinate' in df.columns:
                            df = df[['Index', 'Center X Coordinate', 'Center Y Coordinate']].dropna()

                            # Use x and y coordinates for MSD calculation
                            # Store the (x, y) coordinate pairs of each track in the dictionary
                            self.all_values_dict[track_number] = df.set_index('Index')[['Center X Coordinate', 'Center Y Coordinate']]
                            file_found = True
                            break

                if not file_found:
                    QMessageBox.warning(self, "File Not Found",
                                        f"File track_{track_number}_quantitative_analysis.xlsx does not exist.")
                    return

            # Call MSD plotting function
            self.plot_msd_log_log()
            # timestamp Generate a unique filename based on the timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # format current time
            output_file_name = f'Quant_MSD_{timestamp}.pdf'  # filename contains only timestamp
            output_path = os.path.join(config.quantitative_analysis_output_path, output_file_name)

            try:
                self.canvas.figure.savefig(output_path, dpi=300, bbox_inches='tight')  # save high-resolution image
                QMessageBox.information(self, "Save Successful", f"The line chart has been saved to:\n{output_path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", f"An error occurred while saving the line chart:\n{e}")


        else:
            # Set different colors for different tracks
            colors = itertools.cycle(plt.cm.tab10.colors)
            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)

            # Dictionary to store data for all tracks
            self.all_values_dict = {}

            # Set the maximum value of the x-axis to the number of npy files
            max_frame = self.npy_file_count

            # Search for files in folders one by one
            for track_number in track_numbers:
                track_number = int(track_number)
                file_found = False

                for folder in folders_to_check:
                    file_name = f"track_{track_number}_quantitative_analysis.xlsx"
                    file_path = os.path.join(config.quantitative_analysis_output_path, folder, file_name)

                    if os.path.exists(file_path):
                        # Read Excel file
                        df = pd.read_excel(file_path)

                        # Check if selected column exists
                        if selected_column not in df.columns:
                            QMessageBox.warning(self, "Invalid Data",
                                                f"The selected column {selected_column}  does not exist in the file  {file_name} .")
                            return

                        # Use dropna() to ignore empty values and store track data in the dictionary
                        df = df[['Index', selected_column]].dropna()
                        self.all_values_dict[track_number] = df.set_index('Index')[selected_column]

                        # Plot the track, skipping points with empty values
                        color = next(colors)
                        ax.plot(df['Index'], df[selected_column], marker=None, linestyle='-', color=color, linewidth=1,
                                label=f'Cell{track_number}')

                        file_found = True
                        break


                if not file_found:
                    QMessageBox.warning(self, "File Not Found",
                                        f"File track_{track_number}_quantitative_analysis.xlsx does not exist.")
                    return

            # Create a DataFrame to align data of different lengths
            if self.all_values_dict:
                aligned_df = pd.DataFrame(self.all_values_dict).reindex(range(1, max_frame + 1))  # Reindex to max frames

                # Drop any rows containing NaN (i.e., frames)
                #aligned_df.dropna(axis=0, inplace=True)

                # Calculate the average values for the remaining frames
                average_values = aligned_df.mean(axis=1)

                # Plot the average values curve
                ax.plot(average_values.index, average_values.values, marker=None, linestyle='-', color='red',
                        linewidth=2,
                        label='Mean')

            # Set the x-axis range to the maximum number of frames (i.e., the number of npy files)
            ax.set_xlim(0, max_frame)

            # Ensure the x-axis displays integer ticks
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))



            # Set title and axis labels with consistent font
            ax.set_title(f"{selected_column}", fontsize=16, pad=20, color='#2E2E2E')
            ax.set_xlabel("Frame", fontsize=14, labelpad=12, color='#4D4D4D')

            # Add dashed lines on the y-axis
            ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
            ax.set_facecolor('white')

            # Set bold axis lines
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=10)

            # Add legend to distinguish different tracks and the average
            ax.legend(loc='best', fontsize=10)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file_name = f'Quant_{selected_column}_{timestamp}.pdf'
            output_path = os.path.join(config.quantitative_analysis_output_path, output_file_name)

            try:
                self.canvas.figure.savefig(output_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Save Successful", f"The line chart has been saved to:\n{output_path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", f"An error occurred while saving the line chart:\n{e}")

            # Use dense layout to avoid overlap
            self.canvas.draw()



if __name__ == '__main__':
    # Assume the value of npy_file_count is 100, modify according to actual data
    npy_file_count = 100

    app = QApplication(sys.argv)
    main_window = QuantitativeAnalysisGUI(npy_file_count)
    main_window.show()
    sys.exit(app.exec_())

import sys
import os
import numpy as np
import re
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit,
    QMessageBox, QSizePolicy, QGridLayout, QScrollBar, QHBoxLayout, QPushButton, QSpacerItem)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import config  # Import configuration file
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtGui import QPainter, QPdfWriter
from PyQt5.QtGui import QPageSize


# Custom MplCanvas class with mouse event support
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3.75, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        # Mouse event variables
        self.drag_start = None  # Record the starting point of dragging

        # Save initial range
        self.initial_xlim = (-10, 10)
        self.initial_ylim = (-10, 10)
        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)

        # Bind mouse events
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


class MainWindow(QMainWindow):
    def __init__(self, img_folder_path, npy_folder_path):
        super().__init__()
        self.img_folder = img_folder_path  # Get the image folder path from parameters
        self.npy_folder = npy_folder_path

        self.setWindowTitle('Cell Trajectory Visualization')
        self.setGeometry(100, 100, 1600, 800)  # Adjust window size

        self.track_numbers = []  # Save the list of track numbers for saving images
        self.colors = {}  # Store the mapping between track numbers and colors
        self.canvases = []  # Store all drawn subplots
        self.pages = []  # Store paginated subplots
        self.total_pages = 0

        # Overall layout
        self.layout = QVBoxLayout()

        # Create a horizontal layout to divide content into left and right sections
        self.bottom_layout = QHBoxLayout()

        # Left section: Input box and button section
        self.left_layout = QVBoxLayout()

        # Input box and button section
        input_layout = QHBoxLayout()
        self.label = QLabel("Enter the cell numbers of interest (comma-separated):")
        input_layout.addWidget(self.label)

        self.track_input = QLineEdit()
        self.track_input.setFixedWidth(200)  # Set the width of the input box to 200 pixels
        input_layout.addWidget(self.track_input)

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.plot_track)
        input_layout.addWidget(self.button)

        self.left_layout.addLayout(input_layout)

        # Load data from file to get available track numbers
        self.file_path = os.path.join(config.cell_track_output_path, 'all_cell_tracking')

        # Prefer the specified Excel file
        self.file1_path = os.path.join(self.file_path, 'all_cell_merged_tracking_results.xlsx')

        # Check if the file exists, and use a backup file if it doesn't
        if not os.path.exists(self.file1_path):
            print(f"File not found: {self.file1_path}")
            self.file1_path = os.path.join(self.file_path, 'all_cell_merged_tracking_results.xlsx')
            if not os.path.exists(self.file1_path):
                QMessageBox.warning(self, 'Error', f'File not found: {self.file1_path}')
                return

        # Read track data from the file
        file1_data = pd.read_excel(self.file1_path, sheet_name=None)
        file1_sheet1 = file1_data['Sheet1']

        self.all_track_numbers = file1_sheet1.iloc[:, 0].unique()  # Get unique numbers from the first column

        # Display the image with the smallest number
        self.min_image_canvas = MplCanvas(self, width=4, height=3, dpi=100)

        # Set the size policy of the canvas to make it fill the available space as much as possible
        self.min_image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add the image to the left layout, aligning it to the center and ensuring it occupies more space
        self.left_layout.addWidget(self.min_image_canvas)

        # Create a note label
        self.note_label = QLabel("Note: Different IDs represent different cells.")
        self.note_label.setAlignment(Qt.AlignCenter)  # Center the text

        # Add the note to the left layout
        self.left_layout.addWidget(self.note_label)

        # Add the left layout to the overall layout
        self.bottom_layout.addLayout(self.left_layout, 5)  # Left side takes 5 parts

        # Right section: Display trajectory images
        self.right_layout = QVBoxLayout()

        # Create a layout and widget for displaying images
        self.image_widget = QWidget()
        self.image_layout = QGridLayout()
        self.image_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Set layout alignment
        self.image_widget.setLayout(self.image_layout)
        # Set a fixed size for the image_widget during initialization, e.g., 1200x400

        self.right_layout.addWidget(self.image_widget)

        # Add a scrollbar for pagination
        self.scrollbar = QScrollBar(Qt.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setPageStep(1)
        self.scrollbar.valueChanged.connect(self.update_page)
        self.right_layout.addWidget(self.scrollbar)

        # Add the right layout to the overall layout
        self.bottom_layout.addLayout(self.right_layout, 8)  # Right side takes 8 parts

        # Combine the overall layout
        self.layout.addLayout(self.bottom_layout)  # Add the bottom part to the main layout

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Find and display the image with the smallest number
        self.display_min_image()

    def get_total_frames(self):
        """Get the total number of frames by finding the maximum frame number in image and .npy files"""
        # Extract frame numbers from .npy files
        npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
        npy_frame_numbers = []
        for f in npy_files:
            match = re.search(r'\d+', f)
            if match:
                npy_frame_numbers.append(int(match.group()))

        # Extract frame numbers from image files
        image_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.img'))]
        image_frame_numbers = []
        for f in image_files:
            match = re.findall(r'\d+', f)
            if match:
                image_frame_numbers.append(int(match[0]))  # Assume the first number is the frame number

        # Combine all frame numbers and find the maximum
        all_frame_numbers = npy_frame_numbers + image_frame_numbers
        if all_frame_numbers:
            return max(all_frame_numbers)
        else:
            return 0  # Or you can choose another default value

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

        # Extract numbers from image file names and find the file with the smallest number
        def extract_number(image_name):
            match = re.search(r'\d+', image_name)
            return int(match.group()) if match else float('inf')

        min_image = min(images, key=extract_number)
        min_image_path = os.path.join(image_folder, min_image)

        # Load and display the image
        self.min_image_canvas.ax.clear()  # Clear the current drawing
        img_data = plt.imread(min_image_path)
        self.min_image_canvas.ax.imshow(img_data, aspect='auto')  # Display the image
        self.min_image_canvas.ax.axis('off')  # Hide the axes

        # Ensure the image fills the entire area as much as possible
        self.min_image_canvas.fig.tight_layout(pad=0)  # Reduce the blank area around the image
        self.min_image_canvas.draw()

    def plot_track(self):
        input_text = self.track_input.text()  # Get the track numbers entered by the user
        track_numbers = input_text.split(',')  # Assume the user separates multiple track numbers with commas
        track_numbers = [num.strip() for num in track_numbers if num.strip().isdigit()]

        if not track_numbers:
            QMessageBox.warning(self, 'Error',
                                'Please enter valid cell id. Use commas to separate multiple cell numbers.')
            return

        self.track_numbers = [int(num) for num in track_numbers]  # Convert track numbers to an integer list

        # Check if the track numbers exist
        invalid_tracks = [num for num in self.track_numbers if num not in self.all_track_numbers]
        if invalid_tracks:
            QMessageBox.warning(self, 'Error', f'The following cell numbers do not exist: {invalid_tracks}')
            return

        # Read track data from the file
        file1_data = pd.read_excel(self.file1_path, sheet_name=None)
        file1_sheet1 = file1_data['Sheet1']

        self.track_cell_dict = {}  # Store the list of cell IDs corresponding to each track
        for track_number in self.track_numbers:
            # Filter rows where the first column matches the input track number
            row = file1_sheet1[file1_sheet1.iloc[:, 0] == track_number]

            if row.empty:
                QMessageBox.warning(self, 'Error', f'Cell {track_number} does not exist.')
                continue  # Skip non-existent track numbers

            # Extract data from the row
            row = row.iloc[0]
            cell_ids = row[1:].dropna().values  # Extract all non-empty cell IDs starting from the second column
            self.track_cell_dict[track_number] = cell_ids

        if not self.track_cell_dict:
            QMessageBox.warning(self, 'Error', 'No valid cell ID found.')
            return

        # Get all involved frame numbers
        self.frames = set()
        for cell_ids in self.track_cell_dict.values():
            for cell_id in cell_ids:
                if isinstance(cell_id, str) and 'Cell' in cell_id:
                    try:
                        # Extract the numeric part after 'Cell' as the frame number
                        frame_number = int(cell_id.split('Cell')[1].split('_')[0])  # Get the number after 'Cell'
                        self.frames.add(frame_number)
                    except (ValueError, IndexError):
                        continue
        self.frames = sorted(self.frames)

        # Assign colors to tracks
        cmap = plt.get_cmap('tab10')
        num_colors = len(self.track_numbers)
        colors = cmap.colors * ((num_colors // len(cmap.colors)) + 1)
        colors = colors[:num_colors]
        self.colors = dict(zip(self.track_numbers, colors))

        # Clear previous content
        self.canvases = []
        self.pages = []

        # Plot images for all frames and store them in memory
        self.plot_all_frames()

        # Set the scrollbar
        self.total_pages = len(self.pages)
        self.scrollbar.setMaximum(self.total_pages - 1)
        self.scrollbar.setValue(0)
        self.current_page = 0

        # Update the display
        self.update_page()

    def plot_all_frames(self):
        self.num_cols = 4  # 4 images per row
        self.num_rows = 3  # Maximum of 3 rows per page
        num_images_per_page = self.num_cols * self.num_rows  # Maximum number of images per page

        # Determine the total number of frames k
        k = self.get_total_frames()
        if k == 0:
            QMessageBox.warning(self, 'Error', 'No frames found.')
            return

        # Assign colors to tracks
        cmap = plt.get_cmap('tab10')
        num_colors = len(self.track_numbers)
        colors = cmap.colors * ((num_colors // len(cmap.colors)) + 1)
        colors = colors[:num_colors]
        self.colors = dict(zip(self.track_numbers, colors))

        # Clear previous content
        self.canvases = []
        self.pages = []

        # Iterate through all frames and plot
        for frame_number in range(1, k + 1):
            # Default black background
            base_img = np.zeros((512, 512, 3), dtype=np.uint8)
            outlines = None

            # Find .npy files matching the current frame number
            npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
            matching_files = [f for f in npy_files if
                                re.search(r'\d+', f) and int(re.search(r'\d+', f).group()) == frame_number]

            if matching_files:
                npy_path = os.path.join(self.npy_folder, matching_files[0])
                if os.path.exists(npy_path):
                    dat = np.load(npy_path, allow_pickle=True).item()
                    if 'img' in dat:
                        base_img = dat['img']  # Use the image from the .npy file
                    else:
                        # If 'img' does not exist, find an image in img_folder that matches the frame_number
                        img_file = None
                        for file_name in os.listdir(self.img_folder):
                            # Extract numbers from the image file name
                            img_frame_match = re.findall(r'\d+', file_name)
                            if img_frame_match:  # Check if numbers were found
                                img_frame_number = int(img_frame_match[0])  # Extract the first consecutive number
                                # If the frame number in the image file matches the frame number in the .npy file
                                # and the file is an image format
                                if img_frame_number == frame_number and (
                                        file_name.endswith('.jpeg') or file_name.endswith('.png') or file_name.endswith(
                                    '.jpg')):
                                    img_file = os.path.join(self.img_folder, file_name)
                                    break
                        if img_file is not None:
                            base_img = np.array(Image.open(img_file))
                        else:
                            print(f"Warning: No valid image found for Frame {frame_number}.")

                    if 'outlines' in dat:
                        outlines = dat['outlines']  # Get the outline label map
                    else:
                        print(f"Warning: Frame {frame_number} .npy file has no 'outlines' key.")

            # Process image data to ensure it is 3-channel and of uint8 type
            if len(base_img.shape) == 2:
                base_img = np.stack([base_img] * 3, axis=-1)
            if base_img.dtype != np.uint8:
                base_img = (base_img / base_img.max() * 255).astype(np.uint8)

            img = base_img.copy()

            # Dynamically calculate figsize
            subplot_width = self.image_widget.width() // self.num_cols
            subplot_height = self.image_widget.height() // self.num_rows
            dpi = 100  # Fixed DPI
            figsize_width = subplot_width / dpi
            figsize_height = subplot_height / dpi

            # Create a new Matplotlib canvas
            fig = Figure(figsize=(figsize_width, figsize_height), dpi=dpi)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.set_aspect('equal', 'box')

            # Manually hide other elements of the axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # Set the x-axis label to display the frame number below the image
            ax.set_xlabel(f'Frame {frame_number}', fontsize=8)

            # Draw all track cells in this frame
            if outlines is not None:
                for track_number, cell_ids in self.track_cell_dict.items():
                    for cell_id in cell_ids:
                        if isinstance(cell_id, str) and 'Cell' in cell_id:
                            cell_info = cell_id.split('_')
                            try:
                                # Extract the frame number and cell number
                                cell_frame_number = int(cell_info[0][4:])
                                cell_number = int(cell_info[1])
                            except (ValueError, IndexError):
                                continue

                            # Check if the current frame number matches
                            if cell_frame_number == frame_number:
                                contour = np.argwhere(outlines == cell_number)
                                if len(contour) > 0:
                                    color = self.colors.get(track_number, '#A6CEE3')
                                    ax.plot(contour[:, 1], contour[:, 0], color=color, label=f'Cell {track_number}')

            fig.tight_layout()

            # Add the canvas to the list
            self.canvases.append(canvas)

        # Paginate the canvases
        self.pages = [self.canvases[i:i + num_images_per_page] for i in
                        range(0, len(self.canvases), num_images_per_page)]
        self.total_pages = len(self.pages)

    def update_page(self):
        # Clear the old layout
        for i in reversed(range(self.image_layout.count())):
            widget_to_remove = self.image_layout.itemAt(i).widget()
            self.image_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        if not self.pages:
            return

        current_page_index = self.scrollbar.value()
        current_canvases = self.pages[current_page_index]

        for idx, canvas in enumerate(current_canvases):
            row = idx // self.num_cols  # Use the class attribute num_cols
            col = idx % self.num_cols
            self.image_layout.addWidget(canvas, row, col)  # Directly add to the layout

        # Save the current page
        self.save_current_page(current_page_index)

    def save_current_page(self, page_index):
        track_numbers_str = '_'.join(map(str, self.track_numbers))
        save_directory = os.path.join(config.cell_track_output_path, 'cell_track_images')
        os.makedirs(save_directory, exist_ok=True)

        QApplication.processEvents()
        pixmap = self.image_widget.grab()

        pdf_path = os.path.join(save_directory, f'tracks_{track_numbers_str}_page_{page_index + 1}.pdf')
        pdf_writer = QPdfWriter(pdf_path)

        # Set to A4 size and higher resolution
        pdf_writer.setPageSize(QPageSize(QPageSize.A4))
        pdf_writer.setResolution(300)  # You can also try higher, like 600 DPI

        page_size = QPageSize(QPageSize.A4)
        page_points = page_size.size(QPageSize.Point)
        page_width = page_points.width()
        page_height = page_points.height()

        painter = QPainter(pdf_writer)

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Calculate the scaling factor to fit A4 size
        scale_factor_w = page_width / pixmap_width
        scale_factor_h = page_height / pixmap_height
        scale_factor = min(scale_factor_w, scale_factor_h)

        # Further enlarge based on A4 fit (adjust as needed)
        scale_factor *= 8.0
        painter.scale(scale_factor, scale_factor)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()



def main():
    app = QApplication(sys.argv)
    img_folder_path = config.img_folder_path  # Get the image folder path from the configuration file
    npy_folder_path = config.npy_folder_path  # Get the .npy folder path from the configuration file
    window = MainWindow(img_folder_path, npy_folder_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
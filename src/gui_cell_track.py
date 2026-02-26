import os
import numpy as np
import re
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QLabel,
    QInputDialog, QMessageBox, QComboBox, QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import config  # Ensure there is a config.py file containing cell_track_output_path
import shutil
from PIL import Image

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
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
        if event.inaxes:  # Mouse clicked within canvas
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

class ComparisonMainWindow(QMainWindow):
    def __init__(self, npy_folder, img_folder, tracking_data_path, cell_type):
        super().__init__()
        self.setWindowTitle('Correct Cell Tracking')
        self.setMinimumSize(1200, 600)  # Set minimum window size to allow resizing

        self.npy_folder = npy_folder
        self.img_folder = img_folder
        self.tracking_data_path = tracking_data_path

        # Load the latest tracking data or original data
        self.load_original_tracking_data()

        self.image_files = sorted(
            [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        self.current_image_index = 0

        # Initialize two canvases
        sample_img_file = None

        # Traverse the folder to find the first valid image
        for file_name in os.listdir(self.img_folder):
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):  # Check if file is an image format
                sample_img_file = file_name
                break

        if not sample_img_file:
            raise FileNotFoundError(f"No valid image files were found in the folder:  {self.img_folder}")

        sample_img_path = os.path.join(self.img_folder, sample_img_file)
        sample_img = np.array(Image.open(sample_img_path))  # Load image and convert to numpy array

        # Get image width and height
        height, width = sample_img.shape[:2]

        # Define fixed DPI for display
        display_dpi = 100
        self.canvas_display_width = width / display_dpi  # in inches
        self.canvas_display_height = height / display_dpi  # in inches

        # Initialize two canvases with fixed width and height for display
        self.canvas1 = MplCanvas(self, width=self.canvas_display_width, height=self.canvas_display_height, dpi=display_dpi)
        self.canvas2 = MplCanvas(self, width=self.canvas_display_width, height=self.canvas_display_height, dpi=display_dpi)

        # Set FigureCanvas size policy to Expanding
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Remove axes
        self.canvas1.ax.axis('off')
        self.canvas2.ax.axis('off')

        # Bind click events
        self.canvas1.mpl_connect('button_press_event', self.on_click)
        self.canvas2.mpl_connect('button_press_event', self.on_click)

        # Information labels
        self.info_label1 = QLabel(self)
        self.info_label1.setAlignment(Qt.AlignCenter)
        self.info_label2 = QLabel(self)
        self.info_label2.setAlignment(Qt.AlignCenter)

        # Main layout
        main_layout = QVBoxLayout()

        # Canvas layout
        canvas_layout = QHBoxLayout()
        canvas1_layout = QVBoxLayout()
        canvas2_layout = QVBoxLayout()
        canvas1_layout.addWidget(self.canvas1)
        canvas1_layout.addWidget(self.info_label1)
        canvas2_layout.addWidget(self.canvas2)
        canvas2_layout.addWidget(self.info_label2)
        canvas_layout.addLayout(canvas1_layout)
        canvas_layout.addLayout(canvas2_layout)

        # Create layout for buttons, labels, and dropdown
        button_layout1 = QHBoxLayout()

        # Create "Previous Frame Pair" and "Next Frame Pair" buttons and add to layout
        prev_button = QPushButton('Previous Frame Pair')
        prev_button.clicked.connect(self.show_previous_images)
        prev_button.setFixedWidth(200)  # Adjust button width
        button_layout1.addWidget(prev_button)

        next_button = QPushButton('Next Frame Pair')
        next_button.clicked.connect(self.show_next_images)
        next_button.setFixedWidth(200)
        button_layout1.addWidget(next_button)

        # Add label before input field and button
        frame_label = QLabel('Enter Frame Number for Correction')
        button_layout1.addWidget(frame_label)

        # Change to editable combobox to allow both frame selection and manual input
        self.frame_selector = QComboBox(self)
        self.frame_selector.setEditable(True)  # Allow user to manually enter frame number
        self.frame_selector.setFixedWidth(150)  # Set combobox width
        self.frame_selector.addItems([str(i) for i in range(2, len(self.image_files) + 1)])  # Frame numbers start from 2
        button_layout1.addWidget(self.frame_selector)

        # Create "OK" button
        jump_button = QPushButton('OK')
        jump_button.setFixedWidth(100)
        jump_button.clicked.connect(self.jump_to_frame)
        button_layout1.addWidget(jump_button)

        # Place button_layout1 in a new horizontal layout to center it
        center_layout = QHBoxLayout()
        center_layout.addStretch()  # Left elastic space
        center_layout.addLayout(button_layout1)  # Add button layout to center
        center_layout.addStretch()  # Right elastic space

        # Add to main layout
        main_layout.addLayout(center_layout)  # Only need to add center_layout

        # Add canvas layout and other controls
        main_layout.addLayout(canvas_layout)

        # Set main window
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.correction_ids = {}  # Dictionary to store modified IDs

        # Initialize by displaying the first image pair
        self.update_images()

    def jump_to_frame(self):
        try:
            input_frame_number = int(self.frame_selector.currentText())  # Get user input or selected frame number
            print(input_frame_number)

            if input_frame_number < 2:
                QMessageBox.warning(self, 'Error', 'Please enter a frame number greater than or equal to 2.')
                return

            previous_frame_number = input_frame_number - 1
            current_frame_number = input_frame_number

            # Find files corresponding to frame numbers
            found_prev = False
            found_curr = False
            prev_index = -1
            curr_index = -1

            for i, file_name in enumerate(self.image_files):
                frame_number = int(re.findall(r'\d+', file_name)[0])
                if frame_number == previous_frame_number:
                    prev_index = i
                    found_prev = True
                if frame_number == current_frame_number:
                    curr_index = i
                    found_curr = True

                if found_prev and found_curr:
                    break

            if found_prev and found_curr:
                # Set current_image_index to previous frame index
                self.current_image_index = prev_index
                self.update_images()
            else:
                QMessageBox.warning(self, 'Error',
                                    'The corresponding frame file was not found. Please enter a valid frame number.')
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid numeric frame number.')

    def save_and_reload(self):
        try:
            print('Updating images')
            # self.save_changes()  # First save modified tracking data
            self.reload_images()  # Then reload and update image display
            print('Update complete')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while saving and updating: {str(e)}')

    def reload_images(self):
        try:
            self.update_images()  # Update image display

            # Save the updated two frames
            save_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
            os.makedirs(save_folder, exist_ok=True)

            # Define fixed DPI
            dpi = 100

            # Save image from first canvas
            frame_number1 = int(re.findall(r'\d+', self.image_files[self.current_image_index])[0])
            save_path1 = os.path.join(save_folder, f'cell_track_ture{frame_number1}.png')  # Correct spelling to 'true'
            try:
                # Ensure canvas size consistency
                # Calculate figsize based on current canvas pixel size
                width_px = self.canvas1.width()
                height_px = self.canvas1.height()
                figsize = (width_px / dpi, height_px / dpi)
                self.canvas1.fig.set_size_inches(figsize)
                self.canvas1.fig.savefig(save_path1, dpi=dpi, bbox_inches='tight', pad_inches=0)

                print(f"Frame {frame_number1} image saved to {save_path1}")
            except Exception as e:
                print(f"Error occurred while saving image for frame {frame_number1}: {e}")

            # Save image from second canvas (if exists)
            if self.current_image_index < len(self.image_files) - 1:
                frame_number2 = int(re.findall(r'\d+', self.image_files[self.current_image_index + 1])[0])
                save_path2 = os.path.join(save_folder, f'cell_track_ture{frame_number2}.png')  # Correct spelling to 'true'
                try:
                    # Ensure canvas size consistency
                    width_px = self.canvas2.width()
                    height_px = self.canvas2.height()
                    figsize = (width_px / dpi, height_px / dpi)
                    self.canvas2.fig.set_size_inches(figsize)
                    self.canvas2.fig.savefig(save_path2, dpi=dpi, bbox_inches='tight', pad_inches=0)

                    print(f"Frame {frame_number2} image saved to {save_path2}")
                except Exception as e:
                    print(f"Error occurred while saving image for frame {frame_number2}: {e}")

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while updating the images: {str(e)}')

    def draw_canvas_to_figure(self, canvas, save_ax):
        """Draw current canvas content to another Figure for saving"""
        # Since fig.savefig is used directly, this method is not needed
        pass

    def get_original_image_path(self, image_index):
        """Get original image path based on image index"""
        npy_file = self.image_files[image_index]
        npy_path = os.path.join(self.npy_folder, npy_file)
        dat = np.load(npy_path, allow_pickle=True).item()

        if 'img_path' in dat and dat['img_path']:
            # If 'img_path' exists, return directly
            return dat['img_path']
        else:
            # Find corresponding image from img_folder
            frame_number = int(re.findall(r'\d+', npy_file)[0])
            for file_name in os.listdir(self.img_folder):
                img_frame_match = re.findall(r'\d+', file_name)
                if img_frame_match:
                    img_frame_number = int(img_frame_match[0])
                    if img_frame_number == frame_number and (
                            file_name.endswith('.jpeg') or file_name.endswith('.jpg') or file_name.endswith('.png')):
                        return os.path.join(self.img_folder, file_name)
            raise FileNotFoundError(f"Original image for frame {frame_number} not found.")

    def load_original_tracking_data(self):
        """Load original tracking data file and use directly for modification"""
        self.tracking_data = pd.read_excel(self.tracking_data_path)
        # Save a copy in memory for lookup and reference, this copy will not be modified
        self.original_tracking_data = self.tracking_data.copy()

        # Additional check: ensure 'Index' column exists and is unique
        if 'Index' not in self.tracking_data.columns:
            raise KeyError("The 'Index' column is missing from the tracking data.")

        if not self.tracking_data['Index'].is_unique:
            print("Warning: 'Index' column contains duplicate values.")
            # Depending on requirements, you can choose to remove duplicates or perform other processing
            # For example, remove duplicates:
            # self.tracking_data = self.tracking_data.drop_duplicates(subset=['Index'])
        else:
            print("'Index' column is unique.")

    def update_tracking_data(self, frame_number, old_id, new_id):
        """Update cell ID in tracking data"""
        column_name = f'Frame{frame_number}'

        # Find old ID in original_tracking_data
        original_row_indices = self.original_tracking_data[self.original_tracking_data['Index'] == int(old_id)].index
        print(f"Original Row Indices for old_id={old_id}: {original_row_indices}")
        if original_row_indices.empty:
            QMessageBox.critical(self, 'Error', 'The old ID does not exist in the original data.')
            return

        original_row_index = original_row_indices[0]
        original_cell_id_str = self.original_tracking_data.at[original_row_index, column_name]
        print(f"Original Cell ID String: {original_cell_id_str}")

        # Ensure new ID exists in Index column
        new_id_exists = self.tracking_data['Index'].eq(int(new_id)).any()
        print(f"New ID Exists: {new_id_exists}")
        if not new_id_exists:
            QMessageBox.critical(self, 'Error', 'The new ID does not exist in the index column. Invalid modification.')
            return

        # Find row containing new ID
        new_row_indices = self.tracking_data[self.tracking_data['Index'] == int(new_id)].index
        print(f"New Row Indices for new_id={new_id}: {new_row_indices}")
        if new_row_indices.empty:
            QMessageBox.critical(self, 'Error', 'The new ID does not exist in the index column. Invalid modification.')
            return

        new_row_index = new_row_indices[0]
        print(f"New Row Index: {new_row_index}")

        # Write cell data from old ID to image column of new ID row
        self.tracking_data.at[new_row_index, column_name] = original_cell_id_str
        print(f"Updated tracking_data.at[{new_row_index}, '{column_name}'] to {original_cell_id_str}")

        # Iterate through the column and remove duplicates with same original_cell_id_str elsewhere
        for i, value in self.tracking_data[column_name].items():
            if value == original_cell_id_str and i != new_row_index:
                self.tracking_data.at[i, column_name] = None  # Remove duplicates, set to None or NaN
                print(f"Set tracking_data.at[{i}, '{column_name}'] to None")

        self.tracking_data.to_excel(self.tracking_data_path, index=False)
        print(f"Saved tracking data to {self.tracking_data_path}")

        # Update correction_ids dictionary
        # If old ID exists in correction_ids, delete it
        if f'Cell{frame_number}_{old_id}' in self.correction_ids:
            del self.correction_ids[f'Cell{frame_number}_{old_id}']
            print(f"Deleted correction_ids['Cell{frame_number}_{old_id}']")

        # Add new modification record to correction_ids dictionary
        self.correction_ids[f'Cell{frame_number}_{new_id}'] = new_id
        print(f"Added correction_ids['Cell{frame_number}_{new_id}'] = {new_id}")

        QMessageBox.information(self, 'Success', 'Modification completed successfully.')

    def log_info(self, message, label):
        """Display information in label"""
        label.setText(message)

    def show_previous_images(self):
        """Display previous frame pair"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_images()

    def show_next_images(self):
        """Display next frame pair"""
        if self.current_image_index < len(self.image_files) - 2:
            self.current_image_index += 1
            self.update_images()

    def update_images(self):
        """Update the two currently displayed frames"""
        self.update_canvas(self.canvas1, self.current_image_index, self.info_label1)  # Previous frame
        if self.current_image_index + 1 < len(self.image_files):
            self.update_canvas(self.canvas2, self.current_image_index + 1, self.info_label2)  # Current frame
        else:
            self.canvas2.ax.clear()
            self.canvas2.draw()

    def update_images1(self):
        """Update single canvas image"""
        self.update_canvas(self.canvas1, self.current_image_index, self.info_label1)
        self.canvas2.ax.clear()
        self.canvas2.draw()

    def update_canvas(self, canvas, image_index, info_label):
        """Update image on specified canvas"""
        # Clear canvas
        canvas.ax.clear()
        npy_file = self.image_files[image_index]
        npy_path = os.path.join(self.npy_folder, npy_file)
        frame_number = int(re.findall(r'\d+', npy_file)[0])

        short_npy_name = os.path.basename(npy_path)
        self.log_info(f"{short_npy_name}", info_label)

        if os.path.exists(npy_path):
            # Load .npy file
            dat = np.load(npy_path, allow_pickle=True).item()
            outlines = dat['outlines']  # Outline label image

            if 'img' in dat and dat['img'] is not None:
                base_img = dat['img']  # Use image from npy file
            else:
                # If 'img' does not exist, find image from img_folder that exactly matches frame_number
                img_file = None

                for file_name in os.listdir(self.img_folder):
                    # Extract digits preceding image filename
                    img_frame_match = re.findall(r'\d+', file_name)
                    if img_frame_match:  # Check if digits were found
                        img_frame_number = int(img_frame_match[0])  # Extract first consecutive digits
                        # If frame number in image filename matches frame number in .npy file, and file is .jpeg, .jpg or .png
                        if img_frame_number == frame_number and (
                                file_name.endswith('.jpeg') or file_name.endswith('.jpg') or file_name.endswith('.png')):
                            img_file = os.path.join(self.img_folder, file_name)
                            break

                if img_file is not None:
                    # Open image and convert to numpy array
                    base_img = np.array(Image.open(img_file))
                else:
                    raise FileNotFoundError(f"Image file for frame {frame_number} not found！")

            # If original image is grayscale, convert to RGB format
            if len(base_img.shape) == 2:
                base_img = np.stack([base_img] * 3, axis=-1)

            # Ensure image is uint8 format (avoid data clipping issues)
            if base_img.dtype != np.uint8:
                base_img = (base_img / base_img.max() * 255).astype(np.uint8)

            # Create a copy of RGB image
            img = base_img.copy()

            # Define distinctive outline color (RGB format)
            contour_color = [0, 0, 255]  # Pure blue

            # Iterate through all unique outline IDs and mark contours
            unique_ids = np.unique(outlines)  # Extract unique outline IDs
            for cell_id in unique_ids:
                if cell_id == 0:
                    continue  # Ignore background

                # Use img[outlines == cell_id] = color to mark outlines
                img[outlines == cell_id] = contour_color  # Mark all positions of this cell_id as blue

            # Use imshow to redisplay image
            canvas.ax.imshow(img, aspect='auto')

            # Remove axes and tick marks
            canvas.ax.axis('off')

            # Set image display area to entire canvas
            canvas.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            canvas.ax.set_position([0, 0, 1, 1])  # Remove white borders

            # Refresh canvas to ensure updated image is displayed correctly
            canvas.draw()

            # Calculate centroid position and add text annotations
            for cell_id in unique_ids:
                if cell_id == 0:
                    continue  # Ignore background

                # Calculate cell centroid position
                positions = np.where(outlines == cell_id)
                cy = int(np.mean(positions[0]))  # Average y coordinate
                cx = int(np.mean(positions[1]))  # Average x coordinate

                # Set initial matching ID and draw transparent box at centroid (for later annotation)
                pre_matching_id = str(cell_id)
                canvas.ax.text(cx, cy, pre_matching_id, fontsize=9, color=(0, 0, 0, 0),
                               bbox=dict(facecolor='none', edgecolor='none', alpha=0))

                # Find matching ID
                column_name = f'Frame{frame_number}'
                cell_id_str = f'Cell{frame_number}_{pre_matching_id}'
                matched_id_row = self.tracking_data[self.tracking_data[column_name] == cell_id_str]

                # If match found and row data is not empty, display ID
                if not matched_id_row.empty:
                    matched_id = matched_id_row.iloc[0, 0]  # Get index as matching ID

                    # Check if empty, if so do not display ID
                    if pd.notnull(matched_id):
                        self.correction_ids[cell_id_str] = matched_id

                        # Draw matching ID in image
                        canvas.ax.text(cx, cy, str(matched_id), fontsize=9, color='red',
                                       bbox=dict(facecolor='yellow', alpha=0.5))

            # Refresh canvas again to display text annotations
            canvas.draw()
        else:
            # If .npy file does not exist, show notice
            self.log_info(f"Numpy file not found: {short_npy_name}", info_label)

    def on_click(self, event):
        """Handle click events on canvas to allow users to modify or add cell IDs"""
        if event.inaxes:
            # First check if yellow box is clicked
            for text in event.inaxes.texts:
                contains, _ = text.contains(event)
                bbox_color = text.get_bbox_patch().get_facecolor()

                # Use approximate comparison for colors (floating point)
                if contains and np.allclose(bbox_color[:3], (1.0, 1.0, 0.0)):  # Yellow box logic
                    pre_matching_id = text.get_text()
                    if pre_matching_id.isdigit():
                        frame_number = self.current_image_index + 1 if event.inaxes == self.canvas1.ax else self.current_image_index + 2
                        new_id, ok = QInputDialog.getText(self, 'Edit Cell ID', 'Enter the new Cell ID:')
                        if ok and new_id:
                            # Update tracking data table
                            self.update_tracking_data(frame_number, pre_matching_id, new_id)
                            # Update displayed text
                            text.set_text(new_id)
                            text.set_color('red')
                            text.set_bbox(dict(facecolor='yellow', alpha=0.5))
                            event.inaxes.figure.canvas.draw()

                            # Execute redraw operation
                            self.save_and_reload()

                    return  # Return after processing yellow box, no need to process others

            # Check blue box
            for text in event.inaxes.texts:
                contains, _ = text.contains(event)
                bbox_color = text.get_bbox_patch().get_facecolor()

                if contains and np.allclose(bbox_color[:3], (0.0, 0.0, 0.0)):  # Blue box logic
                    pre_matching_id = text.get_text()
                    if pre_matching_id.isdigit():
                        frame_number = self.current_image_index + 1 if event.inaxes == self.canvas1.ax else self.current_image_index + 2
                        cell_id_str = f'Cell{frame_number}_{pre_matching_id}'

                        # Pop up dialog to get new cell ID
                        new_id, ok = QInputDialog.getText(self, 'Add Cell ID', 'Enter the new Cell ID:')
                        if ok and new_id:
                            # Add new ID to correction_ids dictionary
                            self.correction_ids[cell_id_str] = new_id
                            # Update displayed text
                            text.set_text(new_id)
                            text.set_color('red')
                            text.set_bbox(dict(facecolor='yellow', alpha=0.5))
                            event.inaxes.figure.canvas.draw()  # Redraw image

                            # Find Excel row for new ID and update data
                            try:
                                update_row_index = self.tracking_data[
                                    self.tracking_data['Index'] == int(new_id)].index

                                if not update_row_index.empty:
                                    # Get and print cross-element of row for new_id and column for frame_number
                                    cross_element = self.tracking_data.at[update_row_index[0], f'Frame{frame_number}']
                                    print(f"Cross Element for new_id={new_id}: {cross_element}")

                                    # Update value at this row and column
                                    self.tracking_data.at[update_row_index[0], f'Frame{frame_number}'] = cell_id_str
                                    print(f"Updated tracking_data.at[{update_row_index[0]}, 'Frame{frame_number}'] to {cell_id_str}")

                                    self.tracking_data.to_excel(self.tracking_data_path, index=False)
                                    print(f"Saved tracking data to {self.tracking_data_path}")
                                else:
                                    QMessageBox.warning(self, 'Data Not Found',
                                                        f'ID {new_id} was not found in the updated Excel file.')
                                    print(f"ID {new_id} not found in tracking_data.")
                            except ValueError:
                                QMessageBox.warning(self, 'Invalid ID', 'The new ID must be a valid integer.')
                                print(f"Invalid new_id entered: {new_id}")

                            # Execute redraw operation
                            self.save_and_reload()

                    return  # Return after processing blue box

    def preprocess_images(self, progress_dialog=None, show_progress_bar=None, initial_value=90):
        """Draw and save all images and update total progress bar based on initial value"""
        save_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
        os.makedirs(save_folder, exist_ok=True)

        # Check if folder exists and clear it
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        # Track processed image indices
        drawn_images = set()
        total_files = len(self.image_files)  # Get total number of image files

        # Determine progress step for each image
        progress_step = (100 - initial_value) / total_files

        # Process each file
        for i, npy_file in enumerate(self.image_files):
            if i in drawn_images:
                continue

            self.current_image_index = i
            self.update_images1()  # Update image content

            frame_number = int(re.findall(r'\d+', npy_file)[0])
            save_path = os.path.join(save_folder, f'cell_track_ture{frame_number}.png')
            show_progress_bar(f'Cell tracking image for frame {frame_number} has been successfully generated...')

            # Save current image
            self.canvas1.fig.savefig(save_path)
            # print(f"Frame {frame_number} image saved to {save_path}")

            # Mark current image as processed
            drawn_images.add(i)

            # Update progress bar: incremental update based on initial value
            if progress_dialog:
                progress_value = initial_value + (i + 1) * progress_step
                progress_dialog.setValue(int(progress_value))

        # Set progress bar to 100 when complete
        if progress_dialog:
            progress_dialog.setValue(100)
            progress_dialog.close()
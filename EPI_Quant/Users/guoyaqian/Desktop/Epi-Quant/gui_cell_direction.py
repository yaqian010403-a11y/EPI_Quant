import sys
import os
import re
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox,
    QMessageBox, QPushButton, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PIL import ImageOps  # 确保在文件顶部 import


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


class CellDirection(QMainWindow):
    def __init__(self, img_folder_path, npy_folder_path, celltrack_path, cell_classification_output):
        super().__init__()
        self.img_folder_path = img_folder_path
        self.npy_folder = npy_folder_path
        self.track_excel_path = os.path.join(celltrack_path, 'all_cell_tracking', 'all_cell_merged_tracking_results.xlsx')
        self.cells_info_path = os.path.join(cell_classification_output, 'Cells_info.xlsx')
        self.cell_direction_pictures_path = os.path.join(celltrack_path, 'cell_direction_pictures')
        if not os.path.exists(self.cell_direction_pictures_path):
            os.makedirs(self.cell_direction_pictures_path)

        self.setWindowTitle('Display Cell Displacement Directions')
        self.setGeometry(100, 100, 1000, 800)

        self.layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        input_layout.setSpacing(0)
        input_layout.setAlignment(Qt.AlignCenter)
        input_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.label1 = QLabel("Start Frame:")
        input_layout.addWidget(self.label1)
        self.frame_selector_start = QComboBox()
        self.frame_selector_start.setFixedWidth(100)
        input_layout.addWidget(self.frame_selector_start)

        self.label2 = QLabel("End Frame:")
        input_layout.addWidget(self.label2)
        self.frame_selector_end = QComboBox()
        self.frame_selector_end.setFixedWidth(100)
        input_layout.addWidget(self.frame_selector_end)

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.plot_displacement_arrows)
        input_layout.addWidget(self.button)
        input_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.layout.addLayout(input_layout)

        self.canvas_arrow = MplCanvas(self, width=8, height=4, dpi=100)
        self.canvas_streamline = MplCanvas(self, width=8, height=4, dpi=100)
        self.layout.addWidget(self.canvas_arrow)
        self.layout.addWidget(self.canvas_streamline)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.track_data = pd.read_excel(self.track_excel_path, sheet_name='Sheet1')
        self.cells_info = pd.read_excel(self.cells_info_path, sheet_name='Cell')
        self.populate_frame_selector()

    def populate_frame_selector(self):
        first_row_columns = self.track_data.columns[1:]
        frame_numbers = []
        for column in first_row_columns:
            match = re.search(r'\d+', column)
            if match:
                frame_numbers.append(int(match.group()))
        items = list(map(str, frame_numbers))
        self.frame_selector_start.addItems(items)
        self.frame_selector_end.addItems(items)

    def plot_displacement_arrows(self):
        start_frame = int(self.frame_selector_start.currentText())
        end_frame = int(self.frame_selector_end.currentText())

        column_names = self.track_data.columns[1:]
        start_col = [c for c in column_names if str(start_frame) in c]
        end_col = [c for c in column_names if str(end_frame) in c]
        if not start_col or not end_col:
            QMessageBox.warning(self, 'Error', 'Invalid frame selection.')
            return

        current_column = start_col[0]
        next_column = end_col[0]

        self.canvas_arrow.fig.clear()
        ax_arrow = self.canvas_arrow.fig.add_subplot(111)
        ax_arrow.set_aspect('equal')
        ax_arrow.axis('off')
        self.canvas_arrow.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas_streamline.fig.clear()
        ax_stream = self.canvas_streamline.fig.add_subplot(111)
        ax_stream.set_aspect('equal')
        ax_stream.axis('off')
        self.canvas_streamline.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # === Load background image with inversion ===
        image_path = None
        for fname in os.listdir(self.img_folder_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
            name_no_ext = os.path.splitext(fname)[0]
            match = re.search(r'(\d+)$', name_no_ext)
            if match and int(match.group(1)) == start_frame:
                image_path = os.path.join(self.img_folder_path, fname)
                break

        if image_path:
            try:
                img = Image.open(image_path).convert("L")  # Convert to grayscale
                img = ImageOps.invert(img)  # Invert image for better contrast
                ax_arrow.imshow(img, cmap='gray', origin='upper')
                ax_stream.imshow(img, cmap='gray', origin='upper')
                print(f'Background image loaded with inversion: {image_path}')
            except Exception as e:
                print(f'Failed to load background image: {e}')
        else:
            print(f'No image file found for frame {start_frame}')



        x_list, y_list, u_list, v_list = [], [], [], []
        export_data = []

        for index, row in self.track_data.iterrows():
            current_cell_id = row[current_column]
            next_cell_id = row[next_column]
            if pd.isna(current_cell_id) or pd.isna(next_cell_id):
                continue
            current_center = self.get_cell_center(current_cell_id)
            next_center = self.get_cell_center(next_cell_id)
            if current_center is None or next_center is None:
                continue

            dx = next_center[1] - current_center[1]
            dy = next_center[0] - current_center[0]

            ax_arrow.annotate('',
                              xy=(current_center[1] + dx, current_center[0] + dy),
                              xytext=(current_center[1], current_center[0]),
                              arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5,
                                              mutation_scale=15, shrinkA=0, shrinkB=0))
            ax_stream.annotate('',
                               xy=(current_center[1] + dx, current_center[0] + dy),
                               xytext=(current_center[1], current_center[0]),
                               arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5,
                                               mutation_scale=15, shrinkA=0, shrinkB=0))

            x_list.append(current_center[1])
            y_list.append(current_center[0])
            u_list.append(dx)
            v_list.append(dy)
            export_data.append({
                'Cell ID': current_cell_id,
                'Frame Start': start_frame,
                'Frame End': end_frame,
                'Current Y': current_center[0],
                'Current X': current_center[1],
                'Next Y': next_center[0],
                'Next X': next_center[1],
                'dY': dy,
                'dX': dx
            })

        if x_list and y_list:
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            margin = 20
            for ax_target in [ax_arrow, ax_stream]:
                ax_target.set_xlim(x_min - margin, x_max + margin)
                ax_target.set_ylim(y_max + margin, y_min - margin)

        try:
            from scipy.interpolate import griddata
            x_arr, y_arr, u_arr, v_arr = map(np.array, (x_list, y_list, u_list, v_list))
            xi = np.linspace(min(x_arr), max(x_arr), 100)
            yi = np.linspace(min(y_arr), max(y_arr), 100)
            X, Y = np.meshgrid(xi, yi)
            U = griddata((x_arr, y_arr), u_arr, (X, Y), method='cubic', fill_value=0)
            V = griddata((x_arr, y_arr), v_arr, (X, Y), method='cubic', fill_value=0)
            ax_stream.streamplot(
                X, Y, U, V,
                color='gray',
                linewidth=1.2,
                arrowsize=1.5,
                density=1.2
            )
        except Exception as e:
            print("Streamline plotting failed:", e)

        self.canvas_arrow.draw()
        self.canvas_streamline.draw()

        save_path_arrow = os.path.join(self.cell_direction_pictures_path, f'cell_direction_arrows_only_{start_frame}_{end_frame}.pdf')
        self.canvas_arrow.fig.savefig(save_path_arrow, dpi=300, bbox_inches='tight')

        save_path_stream = os.path.join(self.cell_direction_pictures_path, f'cell_direction_streamline_{start_frame}_{end_frame}.pdf')
        self.canvas_streamline.fig.savefig(save_path_stream, dpi=300, bbox_inches='tight')

        df_export = pd.DataFrame(export_data)
        export_path = os.path.join(self.cell_direction_pictures_path, f'cell_centers_frame_{start_frame}_{end_frame}.xlsx')
        df_export.to_excel(export_path, index=False)

        QMessageBox.information(
            self,
            'Save Successful',
            f'Data and PDFs saved for Frame {start_frame} to {end_frame}.'
        )

    def get_npy_path(self, frame_number):
        npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
        matching_files = [f for f in npy_files if re.search(r'\d+', f) and int(re.search(r'\d+', f).group()) == frame_number]
        if matching_files:
            return os.path.join(self.npy_folder, matching_files[0])
        return None

    def get_cell_center(self, cell_id):
        row = self.cells_info[self.cells_info['Cell Index'] == cell_id]
        if not row.empty:
            x = row['Center X Coordinate'].values[0]
            y = row['Center Y Coordinate'].values[0]
            return (y, x)
        return None
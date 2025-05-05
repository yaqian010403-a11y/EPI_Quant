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
import config  # 确保有一个 config.py 文件，包含 cell_track_output_path
import shutil
from PIL import Image

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        # 鼠标事件变量
        self.drag_start = None  # 记录拖动的起始点

        # 保存初始范围
        self.initial_xlim = (-10, 10)
        self.initial_ylim = (-10, 10)
        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)

        # 绑定鼠标事件
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        if event.inaxes:  # 鼠标在画布内点击
            self.drag_start = (event.xdata, event.ydata)  # 记录起始点

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
        self.setMinimumSize(1200, 600)  # 设置最小窗口尺寸，允许窗口调整

        self.npy_folder = npy_folder
        self.img_folder = img_folder
        self.tracking_data_path = tracking_data_path

        # 加载最新的跟踪数据或原始数据
        self.load_original_tracking_data()

        self.image_files = sorted(
            [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        self.current_image_index = 0

        # 初始化两个画布
        sample_img_file = None

        # 遍历文件夹，找到第一张合法的图片
        for file_name in os.listdir(self.img_folder):
            if file_name.lower().endswith(('.jpeg', '.jpg', '.png')):  # 检查文件是否为图片格式
                sample_img_file = file_name
                break

        if not sample_img_file:
            raise FileNotFoundError(f"No valid image files were found in the folder:  {self.img_folder}")

        sample_img_path = os.path.join(self.img_folder, sample_img_file)
        sample_img = np.array(Image.open(sample_img_path))  # 加载图片并转为 numpy 数组

        # 获取图像的宽度和高度
        height, width = sample_img.shape[:2]

        # 定义固定的 DPI 值用于显示
        display_dpi = 100
        self.canvas_display_width = width / display_dpi  # 英寸
        self.canvas_display_height = height / display_dpi  # 英寸

        # 初始化两个画布，使用固定的宽高用于显示
        self.canvas1 = MplCanvas(self, width=self.canvas_display_width, height=self.canvas_display_height, dpi=display_dpi)
        self.canvas2 = MplCanvas(self, width=self.canvas_display_width, height=self.canvas_display_height, dpi=display_dpi)

        # 设置FigureCanvas的大小策略为Expanding
        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 去除坐标轴
        self.canvas1.ax.axis('off')
        self.canvas2.ax.axis('off')

        # 绑定点击事件
        self.canvas1.mpl_connect('button_press_event', self.on_click)
        self.canvas2.mpl_connect('button_press_event', self.on_click)

        # 信息标签
        self.info_label1 = QLabel(self)
        self.info_label1.setAlignment(Qt.AlignCenter)
        self.info_label2 = QLabel(self)
        self.info_label2.setAlignment(Qt.AlignCenter)

        # 主布局
        main_layout = QVBoxLayout()

        # 画布布局
        canvas_layout = QHBoxLayout()
        canvas1_layout = QVBoxLayout()
        canvas2_layout = QVBoxLayout()
        canvas1_layout.addWidget(self.canvas1)
        canvas1_layout.addWidget(self.info_label1)
        canvas2_layout.addWidget(self.canvas2)
        canvas2_layout.addWidget(self.info_label2)
        canvas_layout.addLayout(canvas1_layout)
        canvas_layout.addLayout(canvas2_layout)

        # 创建按钮、标签和下拉框的布局
        button_layout1 = QHBoxLayout()

        # 创建按钮“上一对帧”和“下一对帧”，并将其添加到布局中
        prev_button = QPushButton('Previous Frame Pair')
        prev_button.clicked.connect(self.show_previous_images)
        prev_button.setFixedWidth(200)  # 调整按钮宽度
        button_layout1.addWidget(prev_button)

        next_button = QPushButton('Next Frame Pair')
        next_button.clicked.connect(self.show_next_images)
        next_button.setFixedWidth(200)
        button_layout1.addWidget(next_button)

        # 添加输入框和按钮前的标签
        frame_label = QLabel('Enter Frame Number for Correction')
        button_layout1.addWidget(frame_label)

        # 修改为可编辑的下拉框，既可以选择帧号，也可以手动输入
        self.frame_selector = QComboBox(self)
        self.frame_selector.setEditable(True)  # 允许用户手动输入帧号
        self.frame_selector.setFixedWidth(150)  # 设置下拉框宽度
        self.frame_selector.addItems([str(i) for i in range(2, len(self.image_files) + 1)])  # 帧号从2开始
        button_layout1.addWidget(self.frame_selector)

        # 创建“确定”按钮
        jump_button = QPushButton('OK')
        jump_button.setFixedWidth(100)
        jump_button.clicked.connect(self.jump_to_frame)
        button_layout1.addWidget(jump_button)

        # 将 button_layout1 放置在一个新的水平布局里，使其居中
        center_layout = QHBoxLayout()
        center_layout.addStretch()  # 左边的弹性空间
        center_layout.addLayout(button_layout1)  # 把按钮布局加到中间
        center_layout.addStretch()  # 右边的弹性空间

        # 添加到主布局
        main_layout.addLayout(center_layout)  # 只需要添加 center_layout

        # 添加画布布局和其他控件
        main_layout.addLayout(canvas_layout)

        # 设置主窗口
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.correction_ids = {}  # 用于存储修改的ID字典

        # 初始化显示第一对图像
        self.update_images()

    def jump_to_frame(self):
        try:
            input_frame_number = int(self.frame_selector.currentText())  # 获取用户输入或选择的帧号
            print(input_frame_number)

            if input_frame_number < 2:
                QMessageBox.warning(self, 'Error', 'Please enter a frame number greater than or equal to 2.')
                return

            previous_frame_number = input_frame_number - 1
            current_frame_number = input_frame_number

            # 查找对应帧号的文件
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
                # 设置 current_image_index 为前一帧的索引
                self.current_image_index = prev_index
                self.update_images()
            else:
                QMessageBox.warning(self, 'Error',
                                    'The corresponding frame file was not found. Please enter a valid frame number.')
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Please enter a valid numeric frame number.')

    def save_and_reload(self):
        try:
            print('更新图像')
            # self.save_changes()  # 先保存修改的跟踪数据
            self.reload_images()  # 然后重新加载并更新图像显示
            print('更新完毕')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while saving and updating: {str(e)}')

    def reload_images(self):
        try:
            self.update_images()  # 更新图像显示

            # 保存更新后的两帧图像
            save_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
            os.makedirs(save_folder, exist_ok=True)

            # 定义固定的 DPI 值
            dpi = 100

            # 保存第一个画布的图像
            frame_number1 = int(re.findall(r'\d+', self.image_files[self.current_image_index])[0])
            save_path1 = os.path.join(save_folder, f'cell_track_ture{frame_number1}.png')  # 修正拼写为 'true'
            try:
                # 确保画布尺寸一致
                # 根据画布的当前像素大小计算figsize
                width_px = self.canvas1.width()
                height_px = self.canvas1.height()
                figsize = (width_px / dpi, height_px / dpi)
                self.canvas1.fig.set_size_inches(figsize)
                self.canvas1.fig.savefig(save_path1, dpi=dpi, bbox_inches='tight', pad_inches=0)

                print(f"第{frame_number1}帧图像已保存至 {save_path1}")
            except Exception as e:
                print(f"Error occurred while saving image for frame {frame_number1}: {e}")

            # 保存第二个画布的图像（如果存在）
            if self.current_image_index < len(self.image_files) - 1:
                frame_number2 = int(re.findall(r'\d+', self.image_files[self.current_image_index + 1])[0])
                save_path2 = os.path.join(save_folder, f'cell_track_ture{frame_number2}.png')  # 修正拼写为 'true'
                try:
                    # 确保画布尺寸一致
                    width_px = self.canvas2.width()
                    height_px = self.canvas2.height()
                    figsize = (width_px / dpi, height_px / dpi)
                    self.canvas2.fig.set_size_inches(figsize)
                    self.canvas2.fig.savefig(save_path2, dpi=dpi, bbox_inches='tight', pad_inches=0)

                    print(f"第{frame_number2}帧图像已保存至 {save_path2}")
                except Exception as e:
                    print(f"Error occurred while saving image for frame {frame_number2}: {e}")

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred while updating the images: {str(e)}')

    def draw_canvas_to_figure(self, canvas, save_ax):
        """将当前画布的内容绘制到另一个 Figure 上，用于保存"""
        # 由于直接使用fig.savefig，不需要此方法
        pass

    def get_original_image_path(self, image_index):
        """根据图像索引获取原始图像的路径"""
        npy_file = self.image_files[image_index]
        npy_path = os.path.join(self.npy_folder, npy_file)
        dat = np.load(npy_path, allow_pickle=True).item()

        if 'img_path' in dat and dat['img_path']:
            # 如果 'img_path' 存在，直接返回
            return dat['img_path']
        else:
            # 从 img_folder 中找到对应的图片
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
        """加载原始的跟踪数据文件，并直接用于修改"""
        self.tracking_data = pd.read_excel(self.tracking_data_path)
        # 保存一份副本在内存中，用于查找和参考，不会修改这个副本
        self.original_tracking_data = self.tracking_data.copy()

        # 额外检查：确保 'Index' 列存在且唯一
        if 'Index' not in self.tracking_data.columns:
            raise KeyError("The 'Index' column is missing from the tracking data.")

        if not self.tracking_data['Index'].is_unique:
            print("Warning: 'Index' column contains duplicate values.")
            # 根据需要，您可以选择删除重复项或进行其他处理
            # 例如，删除重复项：
            # self.tracking_data = self.tracking_data.drop_duplicates(subset=['Index'])
        else:
            print("'Index' column is unique.")

    def update_tracking_data(self, frame_number, old_id, new_id):
        """更新跟踪数据中的细胞ID"""
        column_name = f'Frame{frame_number}'

        # 在 original_tracking_data 中查找旧的 ID
        original_row_indices = self.original_tracking_data[self.original_tracking_data['Index'] == int(old_id)].index
        print(f"Original Row Indices for old_id={old_id}: {original_row_indices}")
        if original_row_indices.empty:
            QMessageBox.critical(self, 'Error', 'The old ID does not exist in the original data.')
            return

        original_row_index = original_row_indices[0]
        original_cell_id_str = self.original_tracking_data.at[original_row_index, column_name]
        print(f"Original Cell ID String: {original_cell_id_str}")

        # 确保新ID存在于序号列中
        new_id_exists = self.tracking_data['Index'].eq(int(new_id)).any()
        print(f"New ID Exists: {new_id_exists}")
        if not new_id_exists:
            QMessageBox.critical(self, 'Error', 'The new ID does not exist in the index column. Invalid modification.')
            return

        # 查找新ID所在行
        new_row_indices = self.tracking_data[self.tracking_data['Index'] == int(new_id)].index
        print(f"New Row Indices for new_id={new_id}: {new_row_indices}")
        if new_row_indices.empty:
            QMessageBox.critical(self, 'Error', 'The new ID does not exist in the index column. Invalid modification.')
            return

        new_row_index = new_row_indices[0]
        print(f"New Row Index: {new_row_index}")

        # 将旧ID对应的细胞数据写到新ID所在行的图像列
        self.tracking_data.at[new_row_index, column_name] = original_cell_id_str
        print(f"Updated tracking_data.at[{new_row_index}, '{column_name}'] to {original_cell_id_str}")

        # 遍历该列，将其他位置中与 original_cell_id_str 相同的元素删除
        for i, value in self.tracking_data[column_name].items():
            if value == original_cell_id_str and i != new_row_index:
                self.tracking_data.at[i, column_name] = None  # 删除重复项，设置为 None 或 NaN
                print(f"Set tracking_data.at[{i}, '{column_name}'] to None")

        self.tracking_data.to_excel(self.tracking_data_path, index=False)
        print(f"Saved tracking data to {self.tracking_data_path}")

        # 更新 correction_ids 字典
        # 如果旧的 correction_ids 中有该旧ID，删除它
        if f'Cell{frame_number}_{old_id}' in self.correction_ids:
            del self.correction_ids[f'Cell{frame_number}_{old_id}']
            print(f"Deleted correction_ids['Cell{frame_number}_{old_id}']")

        # 添加新的修改记录到 correction_ids 字典
        self.correction_ids[f'Cell{frame_number}_{new_id}'] = new_id
        print(f"Added correction_ids['Cell{frame_number}_{new_id}'] = {new_id}")

        QMessageBox.information(self, 'Success', 'Modification completed successfully.')

    def log_info(self, message, label):
        """在标签中显示信息"""
        label.setText(message)

    def show_previous_images(self):
        """显示上一对帧的图像"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_images()

    def show_next_images(self):
        """显示下一对帧的图像"""
        if self.current_image_index < len(self.image_files) - 2:
            self.current_image_index += 1
            self.update_images()

    def update_images(self):
        """更新当前显示的两帧图像"""
        self.update_canvas(self.canvas1, self.current_image_index, self.info_label1)  # 前一帧
        if self.current_image_index + 1 < len(self.image_files):
            self.update_canvas(self.canvas2, self.current_image_index + 1, self.info_label2)  # 当前帧
        else:
            self.canvas2.ax.clear()
            self.canvas2.draw()

    def update_images1(self):
        """更新单个画布的图像"""
        self.update_canvas(self.canvas1, self.current_image_index, self.info_label1)
        self.canvas2.ax.clear()
        self.canvas2.draw()

    def update_canvas(self, canvas, image_index, info_label):
        """更新指定画布上的图像"""
        # 清空画布
        canvas.ax.clear()
        npy_file = self.image_files[image_index]
        npy_path = os.path.join(self.npy_folder, npy_file)
        frame_number = int(re.findall(r'\d+', npy_file)[0])

        short_npy_name = os.path.basename(npy_path)
        self.log_info(f"{short_npy_name}", info_label)

        if os.path.exists(npy_path):
            # 加载 .npy 文件
            dat = np.load(npy_path, allow_pickle=True).item()
            outlines = dat['outlines']  # 轮廓标签图

            if 'img' in dat and dat['img'] is not None:
                base_img = dat['img']  # 使用 npy 文件中的图像
            else:
                # 如果 'img' 不存在，从 img_folder 中找到与 frame_number 完全匹配的图片
                img_file = None

                for file_name in os.listdir(self.img_folder):
                    # 提取图像文件名前的数字
                    img_frame_match = re.findall(r'\d+', file_name)
                    if img_frame_match:  # 检查是否找到了数字
                        img_frame_number = int(img_frame_match[0])  # 提取第一个连续数字
                        # 如果图像文件中的帧号与 .npy 文件中的帧号一致，并且文件是 .jpeg, .jpg 或 .png
                        if img_frame_number == frame_number and (
                                file_name.endswith('.jpeg') or file_name.endswith('.jpg') or file_name.endswith('.png')):
                            img_file = os.path.join(self.img_folder, file_name)
                            break

                if img_file is not None:
                    # 打开图片并转换为 numpy 数组
                    base_img = np.array(Image.open(img_file))
                else:
                    raise FileNotFoundError(f"Image file for frame {frame_number} not found！")

            # 如果原图是灰度图，则转换为 RGB 格式
            if len(base_img.shape) == 2:
                base_img = np.stack([base_img] * 3, axis=-1)

            # 确保图像是 uint8 格式（避免数据裁剪问题）
            if base_img.dtype != np.uint8:
                base_img = (base_img / base_img.max() * 255).astype(np.uint8)

            # 创建一个 RGB 图像的副本
            img = base_img.copy()

            # 定义更显眼的轮廓颜色 (RGB 格式)
            contour_color = [0, 0, 255]  # 纯蓝色

            # 遍历所有唯一的轮廓ID，并标记轮廓
            unique_ids = np.unique(outlines)  # 提取唯一的轮廓 ID
            for cell_id in unique_ids:
                if cell_id == 0:
                    continue  # 忽略背景

                # 使用 img[outlines == cell_id] = color 标记轮廓
                img[outlines == cell_id] = contour_color  # 标记该 cell_id 的所有位置为蓝色

            # 使用 imshow 重新显示图像
            canvas.ax.imshow(img, aspect='auto')

            # 去除坐标轴和刻度线
            canvas.ax.axis('off')

            # 设置图像显示区域为整个画布
            canvas.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            canvas.ax.set_position([0, 0, 1, 1])  # 去除白边

            # 刷新画布，确保更新后的图像被正确显示
            canvas.draw()

            # 计算质心位置并添加文本标注
            for cell_id in unique_ids:
                if cell_id == 0:
                    continue  # 忽略背景

                # 计算细胞质心位置
                positions = np.where(outlines == cell_id)
                cy = int(np.mean(positions[0]))  # y 坐标的平均值
                cx = int(np.mean(positions[1]))  # x 坐标的平均值

                # 设置初始匹配ID，并在质心位置绘制透明框（便于后续标注）
                pre_matching_id = str(cell_id)
                canvas.ax.text(cx, cy, pre_matching_id, fontsize=9, color=(0, 0, 0, 0),
                               bbox=dict(facecolor='none', edgecolor='none', alpha=0))

                # 查找匹配ID
                column_name = f'Frame{frame_number}'
                cell_id_str = f'Cell{frame_number}_{pre_matching_id}'
                matched_id_row = self.tracking_data[self.tracking_data[column_name] == cell_id_str]

                # 如果找到匹配项，且该行数据不为空，才显示ID
                if not matched_id_row.empty:
                    matched_id = matched_id_row.iloc[0, 0]  # 获取索引作为匹配ID

                    # 检查是否为空，如果为空则不显示ID
                    if pd.notnull(matched_id):
                        self.correction_ids[cell_id_str] = matched_id

                        # 在图像中绘制匹配ID
                        canvas.ax.text(cx, cy, str(matched_id), fontsize=9, color='red',
                                       bbox=dict(facecolor='yellow', alpha=0.5))

            # 再次刷新画布以显示文本标注
            canvas.draw()
        else:
            # 如果 .npy 文件不存在，则显示提示信息
            self.log_info(f"Numpy file not found: {short_npy_name}", info_label)

    def on_click(self, event):
        """处理画布上的点击事件，允许用户修改或添加细胞ID"""
        if event.inaxes:
            # 先检查黄色框是否被点击
            for text in event.inaxes.texts:
                contains, _ = text.contains(event)
                bbox_color = text.get_bbox_patch().get_facecolor()

                # 由于颜色是浮点数，进行近似比较
                if contains and np.allclose(bbox_color[:3], (1.0, 1.0, 0.0)):  # 黄色框的处理逻辑
                    pre_matching_id = text.get_text()
                    if pre_matching_id.isdigit():
                        frame_number = self.current_image_index + 1 if event.inaxes == self.canvas1.ax else self.current_image_index + 2
                        new_id, ok = QInputDialog.getText(self, 'Edit Cell ID', 'Enter the new Cell ID:')
                        if ok and new_id:
                            # 更新跟踪数据表格
                            self.update_tracking_data(frame_number, pre_matching_id, new_id)
                            # 更新显示的文本
                            text.set_text(new_id)
                            text.set_color('red')
                            text.set_bbox(dict(facecolor='yellow', alpha=0.5))
                            event.inaxes.figure.canvas.draw()

                            # 执行重新绘图操作
                            self.save_and_reload()

                    return  # 处理黄色框后直接返回，不需要继续处理其他框

            # 检查蓝色框
            for text in event.inaxes.texts:
                contains, _ = text.contains(event)
                bbox_color = text.get_bbox_patch().get_facecolor()

                if contains and np.allclose(bbox_color[:3], (0.0, 0.0, 0.0)):  # 蓝色框的处理逻辑
                    pre_matching_id = text.get_text()
                    if pre_matching_id.isdigit():
                        frame_number = self.current_image_index + 1 if event.inaxes == self.canvas1.ax else self.current_image_index + 2
                        cell_id_str = f'Cell{frame_number}_{pre_matching_id}'

                        # 弹出对话框获取新的细胞ID
                        new_id, ok = QInputDialog.getText(self, 'Add Cell ID', 'Enter the new Cell ID:')
                        if ok and new_id:
                            # 将新 ID 添加到 correction_ids 字典中
                            self.correction_ids[cell_id_str] = new_id
                            # 更新显示的文本
                            text.set_text(new_id)
                            text.set_color('red')
                            text.set_bbox(dict(facecolor='yellow', alpha=0.5))
                            event.inaxes.figure.canvas.draw()  # 重新绘制图像

                            # 查找新ID对应的Excel行，更新数据
                            try:
                                update_row_index = self.tracking_data[
                                    self.tracking_data['Index'] == int(new_id)].index

                                if not update_row_index.empty:
                                    # 获取并打印 new_id 所在行，frame_number 所在列的交叉元素
                                    cross_element = self.tracking_data.at[update_row_index[0], f'Frame{frame_number}']
                                    print(f"Cross Element for new_id={new_id}: {cross_element}")

                                    # 更新该行、该列的值
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

                            # 执行重新绘图操作
                            self.save_and_reload()

                    return  # 处理蓝色框后直接返回

    def preprocess_images(self, progress_dialog=None, show_progress_bar=None, initial_value=90):
        """绘制并保存所有的图像，并根据初始进度值更新总进度条"""
        save_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
        os.makedirs(save_folder, exist_ok=True)

        # 检查文件夹是否存在，并清空
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        # 用于跟踪已绘制的图像索引
        drawn_images = set()
        total_files = len(self.image_files)  # 获取所有图像文件的数量

        # 确定进度条的每个图像处理比例
        progress_step = (100 - initial_value) / total_files

        # 遍历每个文件进行处理
        for i, npy_file in enumerate(self.image_files):
            if i in drawn_images:
                continue

            self.current_image_index = i
            self.update_images1()  # 更新图像内容

            frame_number = int(re.findall(r'\d+', npy_file)[0])
            save_path = os.path.join(save_folder, f'cell_track_ture{frame_number}.png')
            show_progress_bar(f'Cell tracking image for frame {frame_number} has been successfully generated...')

            # 保存当前图像
            self.canvas1.fig.savefig(save_path)
            # print(f"第{frame_number}帧图像已保存至 {save_path}")

            # 标记当前图像为已绘制
            drawn_images.add(i)

            # 更新进度条：基于初始值的增量更新
            if progress_dialog:
                progress_value = initial_value + (i + 1) * progress_step
                progress_dialog.setValue(int(progress_value))

        # 处理完成后将进度条设置为 100
        if progress_dialog:
            progress_dialog.setValue(100)
            progress_dialog.close()
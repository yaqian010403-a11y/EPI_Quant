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
import config  # 引入配置文件
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtGui import QPainter, QPdfWriter
from PyQt5.QtGui import QPageSize


# 自定义 MplCanvas 类，加入鼠标事件支持
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3.75, dpi=100):
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


class MainWindow(QMainWindow):
    def __init__(self, img_folder_path, npy_folder_path):
        super().__init__()
        self.img_folder = img_folder_path  # 从参数获取图像文件夹路径
        self.npy_folder = npy_folder_path

        self.setWindowTitle('Cell Trajectory Visualization')
        self.setGeometry(100, 100, 1600, 800)  # 调整窗口大小

        self.track_numbers = []  # 保存轨迹号列表以便保存图像使用
        self.colors = {}  # 用于存储轨迹编号与颜色的映射
        self.canvases = []  # 存储所有绘制的子图
        self.pages = []  # 存储分页后的子图
        self.total_pages = 0

        # 整体布局
        self.layout = QVBoxLayout()

        # 创建一个水平布局，将内容分成左右两边
        self.bottom_layout = QHBoxLayout()

        # 左侧部分：输入框和按钮部分
        self.left_layout = QVBoxLayout()

        # 输入框和按钮部分
        input_layout = QHBoxLayout()
        self.label = QLabel("Enter the cell numbers of interest (comma-separated):")
        input_layout.addWidget(self.label)

        self.track_input = QLineEdit()
        self.track_input.setFixedWidth(200)  # 设置输入框的宽度为200像素
        input_layout.addWidget(self.track_input)

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.plot_track)
        input_layout.addWidget(self.button)

        self.left_layout.addLayout(input_layout)

        # 从文件加载数据，获取可用的轨迹编号
        self.file_path = os.path.join(config.cell_track_output_path, 'all_cell_tracking')

        # 优先选择指定的 Excel 文件
        self.file1_path = os.path.join(self.file_path, 'all_cell_merged_tracking_results.xlsx')

        # 检查文件是否存在，若不存在则使用备用文件
        if not os.path.exists(self.file1_path):
            print(f"未找到文件: {self.file1_path}")
            self.file1_path = os.path.join(self.file_path, 'all_cell_merged_tracking_results.xlsx')
            if not os.path.exists(self.file1_path):
                QMessageBox.warning(self, 'Error', f'File not found: {self.file1_path}')
                return

        # 读取文件中的轨迹数据
        file1_data = pd.read_excel(self.file1_path, sheet_name=None)
        file1_sheet1 = file1_data['Sheet1']

        self.all_track_numbers = file1_sheet1.iloc[:, 0].unique()  # 获取第一列的唯一数字

        # 显示最小编号的图片
        self.min_image_canvas = MplCanvas(self, width=4, height=3, dpi=100)

        # 设置画布的尺寸策略，让它尽可能填充可用空间
        self.min_image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 将图片按中心对齐，并确保其占据更多空间
        self.left_layout.addWidget(self.min_image_canvas)

        # 创建注释标签
        self.note_label = QLabel("Note: Different IDs represent different cells.")
        self.note_label.setAlignment(Qt.AlignCenter)  # 设置文字居中

        # 将注释添加到左侧布局
        self.left_layout.addWidget(self.note_label)

        # 将左侧布局添加到整体布局中
        self.bottom_layout.addLayout(self.left_layout, 5)  # 左边占5份

        # 右侧部分：轨迹图像显示
        self.right_layout = QVBoxLayout()

        # 创建一个用于显示图像的布局和小部件
        self.image_widget = QWidget()
        self.image_layout = QGridLayout()
        self.image_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 设置布局对齐方式
        self.image_widget.setLayout(self.image_layout)
        # 在初始化中为 image_widget 设置固定大小，比如 1200x400

        self.right_layout.addWidget(self.image_widget)

        # 添加滚动条用于翻页
        self.scrollbar = QScrollBar(Qt.Horizontal)
        self.scrollbar.setMinimum(0)
        self.scrollbar.setPageStep(1)
        self.scrollbar.valueChanged.connect(self.update_page)
        self.right_layout.addWidget(self.scrollbar)

        # 将右侧布局添加到整体布局
        self.bottom_layout.addLayout(self.right_layout, 8)  # 右边占8份

        # 将整体布局组合
        self.layout.addLayout(self.bottom_layout)  # 下半部分添加到主布局中

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # 查找并显示最小编号的图片
        self.display_min_image()

    def get_total_frames(self):
        """获取图像和.npy文件中的最大帧号作为总帧数"""
        # 从.npy文件中提取帧号
        npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
        npy_frame_numbers = []
        for f in npy_files:
            match = re.search(r'\d+', f)
            if match:
                npy_frame_numbers.append(int(match.group()))

        # 从图像文件中提取帧号
        image_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.img'))]
        image_frame_numbers = []
        for f in image_files:
            match = re.findall(r'\d+', f)
            if match:
                image_frame_numbers.append(int(match[0]))  # 假设第一个数字是帧号

        # 合并所有帧号并找到最大值
        all_frame_numbers = npy_frame_numbers + image_frame_numbers
        if all_frame_numbers:
            return max(all_frame_numbers)
        else:
            return 0  # 或者您可以选择其他默认值

    def display_min_image(self):
        """查找并显示 cell_track_output_pictures 文件夹中编号最小的图片，支持鼠标交互"""
        image_folder = os.path.join(config.cell_track_output_path, 'cell_track_output_pictures')
        if not os.path.exists(image_folder):
            QMessageBox.warning(self, 'Error', f'The directory {image_folder} does not exist.')
            return

        # 获取所有图片文件
        images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.img'))]
        if not images:
            QMessageBox.warning(self, 'Error', 'No images were found.')
            return

        # 提取图片文件名中的数字，找到数字最小的文件
        def extract_number(image_name):
            match = re.search(r'\d+', image_name)
            return int(match.group()) if match else float('inf')

        min_image = min(images, key=extract_number)
        min_image_path = os.path.join(image_folder, min_image)

        # 加载并显示图片
        self.min_image_canvas.ax.clear()  # 清除当前绘制
        img_data = plt.imread(min_image_path)
        self.min_image_canvas.ax.imshow(img_data, aspect='auto')  # 显示图像
        self.min_image_canvas.ax.axis('off')  # 隐藏坐标轴

        # 确保图像尽可能填充整个区域
        self.min_image_canvas.fig.tight_layout(pad=0)  # 减少图像周围的空白区域
        self.min_image_canvas.draw()

    def plot_track(self):
        input_text = self.track_input.text()  # 获取用户输入的轨迹序号
        track_numbers = input_text.split(',')  # 假设用户用逗号分隔多个轨迹编号
        track_numbers = [num.strip() for num in track_numbers if num.strip().isdigit()]

        if not track_numbers:
            QMessageBox.warning(self, 'Error',
                                'Please enter valid cell id. Use commas to separate multiple cell numbers.')
            return

        self.track_numbers = [int(num) for num in track_numbers]  # 将轨迹编号转换为整数列表

        # 检查轨迹编号是否存在
        invalid_tracks = [num for num in self.track_numbers if num not in self.all_track_numbers]
        if invalid_tracks:
            QMessageBox.warning(self, 'Error', f'The following cell numbers do not exist: {invalid_tracks}')
            return

        # 读取文件中的轨迹数据
        file1_data = pd.read_excel(self.file1_path, sheet_name=None)
        file1_sheet1 = file1_data['Sheet1']

        self.track_cell_dict = {}  # 存储每个轨迹对应的细胞ID列表
        for track_number in self.track_numbers:
            # 筛选出第一列值为输入轨迹序号的行
            row = file1_sheet1[file1_sheet1.iloc[:, 0] == track_number]

            if row.empty:
                QMessageBox.warning(self, 'Error', f'Cell {track_number} does not exist.')
                continue  # 跳过不存在的轨迹编号

            # 提取该行数据
            row = row.iloc[0]
            cell_ids = row[1:].dropna().values  # 提取从第二列开始的所有非空细胞ID
            self.track_cell_dict[track_number] = cell_ids

        if not self.track_cell_dict:
            QMessageBox.warning(self, 'Error', 'No valid cell ID found.')
            return

        # 获取所有涉及的帧号
        self.frames = set()
        for cell_ids in self.track_cell_dict.values():
            for cell_id in cell_ids:
                if isinstance(cell_id, str) and 'Cell' in cell_id:
                    try:
                        # 提取 'Cell' 后面的数字部分作为帧号
                        frame_number = int(cell_id.split('Cell')[1].split('_')[0])  # 获取 Cell 后面的数字部分
                        self.frames.add(frame_number)
                    except (ValueError, IndexError):
                        continue
        self.frames = sorted(self.frames)

        # 为轨迹分配颜色
        cmap = plt.get_cmap('tab10')
        num_colors = len(self.track_numbers)
        colors = cmap.colors * ((num_colors // len(cmap.colors)) + 1)
        colors = colors[:num_colors]
        self.colors = dict(zip(self.track_numbers, colors))

        # 清空之前的内容
        self.canvases = []
        self.pages = []

        # 绘制所有帧的图像并存储在内存中
        self.plot_all_frames()

        # 设置滚动条
        self.total_pages = len(self.pages)
        self.scrollbar.setMaximum(self.total_pages - 1)
        self.scrollbar.setValue(0)
        self.current_page = 0

        # 更新显示
        self.update_page()

    def plot_all_frames(self):
        self.num_cols = 4  # 每行4个图
        self.num_rows = 3  # 每页最多显示3行
        num_images_per_page = self.num_cols * self.num_rows  # 每页最多显示的图片数

        # 确定总帧数 k
        k = self.get_total_frames()
        if k == 0:
            QMessageBox.warning(self, 'Error', 'No frames found.')
            return

        # 为轨迹分配颜色
        cmap = plt.get_cmap('tab10')
        num_colors = len(self.track_numbers)
        colors = cmap.colors * ((num_colors // len(cmap.colors)) + 1)
        colors = colors[:num_colors]
        self.colors = dict(zip(self.track_numbers, colors))

        # 清空之前的内容
        self.canvases = []
        self.pages = []

        # 遍历所有帧并绘制
        for frame_number in range(1, k + 1):
            # 默认黑色背景
            base_img = np.zeros((512, 512, 3), dtype=np.uint8)
            outlines = None

            # 查找与当前帧号匹配的 .npy 文件
            npy_files = [f for f in os.listdir(self.npy_folder) if f.endswith('.npy')]
            matching_files = [f for f in npy_files if
                              re.search(r'\d+', f) and int(re.search(r'\d+', f).group()) == frame_number]

            if matching_files:
                npy_path = os.path.join(self.npy_folder, matching_files[0])
                if os.path.exists(npy_path):
                    dat = np.load(npy_path, allow_pickle=True).item()
                    if 'img' in dat:
                        base_img = dat['img']  # 使用 npy 文件中的图像
                    else:
                        # 如果 'img' 不存在，从 img_folder 中找到与 frame_number 完全匹配的图片
                        img_file = None
                        for file_name in os.listdir(self.img_folder):
                            # 提取图像文件名前的数字
                            img_frame_match = re.findall(r'\d+', file_name)
                            if img_frame_match:  # 检查是否找到了数字
                                img_frame_number = int(img_frame_match[0])  # 提取第一个连续数字
                                # 如果图像文件中的帧号与 .npy 文件中的帧号一致，并且文件是图像格式
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
                        outlines = dat['outlines']  # 获取轮廓标签图
                    else:
                        print(f"Warning: Frame {frame_number} .npy file has no 'outlines' key.")

            # 处理图像数据，确保为3通道且为uint8类型
            if len(base_img.shape) == 2:
                base_img = np.stack([base_img] * 3, axis=-1)
            if base_img.dtype != np.uint8:
                base_img = (base_img / base_img.max() * 255).astype(np.uint8)

            img = base_img.copy()

            # 动态计算 figsize
            subplot_width = self.image_widget.width() // self.num_cols
            subplot_height = self.image_widget.height() // self.num_rows
            dpi = 100  # 固定DPI
            figsize_width = subplot_width / dpi
            figsize_height = subplot_height / dpi

            # 创建一个新的 Matplotlib 画布
            fig = Figure(figsize=(figsize_width, figsize_height), dpi=dpi)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.set_aspect('equal', 'box')

            # 手动隐藏轴的其他元素
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # 设置 x 轴标签，将帧号显示在图像下方
            ax.set_xlabel(f'Frame {frame_number}', fontsize=8)

            # 在此帧中绘制所有轨迹的细胞
            if outlines is not None:
                for track_number, cell_ids in self.track_cell_dict.items():
                    for cell_id in cell_ids:
                        if isinstance(cell_id, str) and 'Cell' in cell_id:
                            cell_info = cell_id.split('_')
                            try:
                                # 提取帧号和细胞编号
                                cell_frame_number = int(cell_info[0][4:])
                                cell_number = int(cell_info[1])
                            except (ValueError, IndexError):
                                continue

                            # 判断当前帧号是否匹配
                            if cell_frame_number == frame_number:
                                contour = np.argwhere(outlines == cell_number)
                                if len(contour) > 0:
                                    color = self.colors.get(track_number, '#A6CEE3')
                                    ax.plot(contour[:, 1], contour[:, 0], color=color, label=f'Cell {track_number}')

            fig.tight_layout()

            # 将 canvas 添加到列表中
            self.canvases.append(canvas)

        # 将 canvases 分页
        self.pages = [self.canvases[i:i + num_images_per_page] for i in
                      range(0, len(self.canvases), num_images_per_page)]
        self.total_pages = len(self.pages)

    def update_page(self):
        # 清空旧布局
        for i in reversed(range(self.image_layout.count())):
            widget_to_remove = self.image_layout.itemAt(i).widget()
            self.image_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        if not self.pages:
            return

        current_page_index = self.scrollbar.value()
        current_canvases = self.pages[current_page_index]

        for idx, canvas in enumerate(current_canvases):
            row = idx // self.num_cols  # 使用类属性 num_cols
            col = idx % self.num_cols
            self.image_layout.addWidget(canvas, row, col)  # 直接添加到布局

        # 保存当前页面
        self.save_current_page(current_page_index)

    def save_current_page(self, page_index):
        track_numbers_str = '_'.join(map(str, self.track_numbers))
        save_directory = os.path.join(config.cell_track_output_path, 'cell_track_images')
        os.makedirs(save_directory, exist_ok=True)

        QApplication.processEvents()
        pixmap = self.image_widget.grab()

        pdf_path = os.path.join(save_directory, f'tracks_{track_numbers_str}_page_{page_index + 1}.pdf')
        pdf_writer = QPdfWriter(pdf_path)

        # 设置为A4大小和更高分辨率
        pdf_writer.setPageSize(QPageSize(QPageSize.A4))
        pdf_writer.setResolution(300)  # 你也可以尝试更高，比如600 DPI

        page_size = QPageSize(QPageSize.A4)
        page_points = page_size.size(QPageSize.Point)
        page_width = page_points.width()
        page_height = page_points.height()

        painter = QPainter(pdf_writer)

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # 计算适配A4大小的缩放因子
        scale_factor_w = page_width / pixmap_width
        scale_factor_h = page_height / pixmap_height
        scale_factor = min(scale_factor_w, scale_factor_h)

        # 在适配A4的基础上再放大一倍（可根据需要调整）
        scale_factor *= 8.0
        painter.scale(scale_factor, scale_factor)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()


def main():
    app = QApplication(sys.argv)
    img_folder_path = config.img_folder_path  # 从配置文件获取图像文件夹路径
    npy_folder_path = config.npy_folder_path  # 确保从配置文件获取.npy文件夹路径
    window = MainWindow(img_folder_path, npy_folder_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
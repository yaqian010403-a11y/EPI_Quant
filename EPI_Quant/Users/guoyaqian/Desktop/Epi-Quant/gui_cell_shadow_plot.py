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
from datetime import datetime  # 添加导入
import config  # 确保你有一个config.py文件，其中定义了所需的路径
from scipy.stats import linregress

# 自动根据操作系统选择字体路径
if sys.platform.startswith('win'):
    font_path = "C:\\Windows\\Fonts\\simhei.ttf"  # Windows 系统
elif sys.platform.startswith('darwin'):
    font_path = "/System/Library/Fonts/PingFang.ttc"  # macOS 系统
else:
    font_path = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"  # Linux

# 检查字体文件是否存在
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.family'] = 'sans-serif'

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
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

        # 隐藏坐标轴
        self.ax.axis('off')  # 添加这一行

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


class CellShadowLineGUI(QMainWindow):
    def __init__(self, quantitative_analysis_output_path, all_cell_quantitative_analysis_output, cell_track_output_pictures):
        super().__init__()
        self.setWindowTitle("Multi-Cell Quantitative Analysis")
        self.setGeometry(100, 100, 1300, 600)
        self.category_inputs = []  # 用于存储动态添加的类别输入框
        self.base_path = all_cell_quantitative_analysis_output
        self.image_folder = cell_track_output_pictures
        self.quantitative_analysis_output_path = quantitative_analysis_output_path

        # 可用特征
        self.features = [
            'Cell Area', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio',
            'Cell Perimeter', 'Approximate Polygon Vertex Count',
            'Fitted Ellipse Minor Axis', 'Fitted Ellipse Major Axis', 'Ellipse Major/Minor Axis Ratio',
            'Fitted Ellipse Angle', 'Circumscribed Circle Radius',
            'Inscribed Circle Radius', 'Center X Coordinate', 'Center Y Coordinate', 'Cell Left Boundary',
            'Cell Right Boundary', 'Circularity', 'P-Value', 'leading_edge', 'γ'
        ]

        # 主布局
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.bottom_layout = QHBoxLayout()

        # 左侧部分
        self.left_layout = QVBoxLayout()

        # 特征选择、时间间隔输入和添加类别按钮布局
        feature_layout = QHBoxLayout()
        self.feature_label = QLabel("Select Feature:")
        self.feature_combo = QComboBox(self)
        self.feature_combo.setFixedWidth(100)  # 设置下拉框宽度为100像素
        self.feature_combo.addItems(self.features)

        self.time_interval_label = QLabel("Time Interval (s):")
        self.time_interval_spinbox = QSpinBox(self)
        self.time_interval_spinbox.setRange(1, 100000)  # 设置范围
        self.time_interval_spinbox.setValue(60)  # 默认值为60秒
        self.time_interval_spinbox.setFixedWidth(50)

        self.add_category_input_button = QPushButton("Add Group", self)
        self.add_category_input_button.setFixedWidth(100)  # 设置按钮宽度为100像素

        self.add_category_input_button.clicked.connect(self.add_category_input)

        # 将特征选择、时间间隔输入和按钮添加到水平布局
        feature_layout.addWidget(self.feature_label)
        feature_layout.addWidget(self.feature_combo)
        feature_layout.addWidget(self.time_interval_label)
        feature_layout.addWidget(self.time_interval_spinbox)
        feature_layout.addWidget(self.add_category_input_button)

        self.left_layout.addLayout(feature_layout)

        # 输入部分（滚动区域支持动态添加）
        scroll_area = QScrollArea(self)
        self.form_widget = QWidget()

        # 使用 QGridLayout 作为布局
        self.form_layout = QGridLayout(self.form_widget)
        self.form_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 确保从左上角开始排列

        # 设置水平和垂直间距为0
        self.form_layout.setHorizontalSpacing(5)
        self.form_layout.setVerticalSpacing(5)

        # 移除布局的内容边距
        self.form_layout.setContentsMargins(5, 5, 5, 5)

        self.form_widget.setLayout(self.form_layout)
        scroll_area.setWidget(self.form_widget)
        scroll_area.setWidgetResizable(True)
        self.left_layout.addWidget(scroll_area)

        # 默认添加类别1
        self.add_category_input(is_default=True)

        # 绘制按钮
        self.plot_button = QPushButton("Plot", self)
        self.plot_button.clicked.connect(self.plot_selected_feature)
        self.left_layout.addWidget(self.plot_button)

        # 显示最小编号的图片
        self.min_image_canvas = MplCanvas(self, width=4, height=4, dpi=100)
        self.min_image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_layout.addWidget(self.min_image_canvas)

        # 添加左侧布局
        self.bottom_layout.addLayout(self.left_layout, 5)

        # 右侧部分（显示图表）
        self.right_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.right_layout.addWidget(self.canvas)
        self.bottom_layout.addLayout(self.right_layout, 8)

        # 将左右布局添加到主布局
        main_layout.addLayout(self.bottom_layout)

        # 显示图片
        self.display_min_image()

    def add_category_input(self, is_default=False):
        """动态添加输入框，并将其从左上角开始排列"""
        # 创建输入框
        input_line = QLineEdit(self)
        input_line.setPlaceholderText("Enter Cell ID (comma-separated):")
        input_line.setFixedWidth(155)  # 输入框宽度固定为155像素

        self.category_inputs.append(input_line)

        # 动态计算行和列
        row = (len(self.category_inputs) - 1) // 2  # 每行显示两个类别
        col = (len(self.category_inputs) - 1) % 2  # 当前列号：0 或 1

        # 默认类别编号
        category_number = 1 if is_default else len(self.category_inputs)

        # 创建类别标签
        category_label = QLabel(f"Group {category_number}:")
        category_label.setFixedWidth(60)  # 设置标签固定宽度

        # 移除标签和输入框的外边距和内边距
        category_label.setStyleSheet("margin: 0px; padding: 0px;")
        input_line.setStyleSheet("margin: 0px; padding: 0px;")

        # 添加控件到网格布局
        self.form_layout.addWidget(category_label, row, col * 2, alignment=Qt.AlignLeft)  # 标签占左列
        self.form_layout.addWidget(input_line, row, col * 2 + 1, alignment=Qt.AlignLeft)  # 输入框占右列

    def load_track_data(self, track_numbers, feature):
        """根据轨迹编号加载 Excel 数据并返回指定特征的 DataFrame"""
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
        """计算每个时间点的平均值和标准差"""
        mean_values = data_frame.mean(axis=1)
        std_values = data_frame.std(axis=1)
        return mean_values, std_values

    def generate_soft_colors(self, num_colors):
        if num_colors <= 1:
            return [(0.121, 0.466, 0.705, 1)]  # matplotlib 默认蓝色

        # 使用 Set2 调色板，生成对比度更高的颜色
        cmap = colormaps['Set2']  # 获取 Set2 调色板
        if num_colors > 8:
            # 如果需要的颜色超过 8 个，可以扩展颜色或选择不同的调色板
            cmap = colormaps['tab20']  # 使用更大的 tab20 调色板
        colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
        return colors

    def plot_selected_feature(self):
        """根据选择的特征决定调用哪个绘图函数。"""
        selected_feature = self.feature_combo.currentText()
        self.canvas.ax.clear()  # 清除之前的绘图

        if selected_feature == 'γ':
            self.plot_msd_boxplot()
        else:
            self.plot_lines()

    def plot_lines(self):
        """绘制优化后的带矩形边界的折线图"""
        self.canvas.ax.clear()

        # 获取用户选择的特征
        selected_feature = self.feature_combo.currentText()

        # 动态生成颜色
        num_categories = len(self.category_inputs)
        colors = self.generate_soft_colors(num_categories)

        all_mean_values = []  # 收集所有平均值范围，用于确定统一的范围
        all_time_points = []

        for idx, input_line in enumerate(self.category_inputs):
            track_numbers = [int(x.strip()) for x in input_line.text().split(',') if x.strip().isdigit()]
            if not track_numbers:
                continue

            # 加载数据
            data_frame = self.load_track_data(track_numbers, selected_feature)
            if data_frame is None:
                continue

            # 计算平均值和标准差
            mean_values, std_values = self.calculate_mean_std(data_frame)

            # 收集时间点和平均值
            time_points = range(1, len(mean_values) + 1)
            all_mean_values.extend(mean_values)
            all_time_points.extend(time_points)

            # 绘制带阴影的折线图
            color = colors[idx]  # 从生成的颜色列表中取颜色
            self.canvas.ax.plot(time_points, mean_values, label=f"Group {idx + 1}", color=color, linewidth=2.5)
            self.canvas.ax.fill_between(time_points, mean_values - std_values, mean_values + std_values,
                                        color=color, alpha=0.3)

        if not all_mean_values or not all_time_points:
            QMessageBox.warning(self, "No Data", "No data available to plot.")
            return

        # 确保包含原点
        x_min, x_max = min(all_time_points, default=0), max(all_time_points, default=1)
        y_min, y_max = min(all_mean_values, default=0), max(all_mean_values, default=1)

        # 动态增加边界留白
        x_margin = 0.05 * (x_max - x_min) if x_max != x_min else 1
        y_margin = 0.05 * (y_max - y_min) if y_max != y_min else 1

        # 设置坐标轴范围
        self.canvas.ax.set_xlim(x_min - x_margin, x_max + x_margin)
        self.canvas.ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # 设置完整矩形边框
        for spine in ['top', 'bottom', 'left', 'right']:
            self.canvas.ax.spines[spine].set_visible(True)
            self.canvas.ax.spines[spine].set_color('#4D4D4D')  # 边框颜色
            self.canvas.ax.spines[spine].set_linewidth(1.2)  # 边框线宽

        # 将轴的位置设置为默认位置，避免轴移动到数据内部
        self.canvas.ax.spines['left'].set_position(('outward', 0))
        self.canvas.ax.spines['bottom'].set_position(('outward', 0))
        self.canvas.ax.spines['right'].set_position(('outward', 0))
        self.canvas.ax.spines['top'].set_position(('outward', 0))

        # 设置 x 轴和 y 轴的刻度仅在底部和左侧显示，顶部和右侧不显示刻度
        self.canvas.ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, direction='in')
        self.canvas.ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, direction='in')

        # 设置图表信息
        self.canvas.ax.set_title(f"{selected_feature}", fontsize=16, pad=20, color='#2E2E2E')
        self.canvas.ax.set_xlabel("Frame", fontsize=14, labelpad=12, color='#4D4D4D')

        # 调整图例
        self.canvas.ax.legend(fontsize=12, loc='upper right', frameon=False)

        # 美化刻度
        self.canvas.ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, color='#4D4D4D')

        # 绘制图像
        self.canvas.draw()

        # 保存图像到指定路径
        save_dir = self.quantitative_analysis_output_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化时间，例如 20250110_123456
        save_filename = f"mean_with_variance_{current_time}.pdf"
        save_path = os.path.join(save_dir, save_filename)

        try:
            self.canvas.fig.savefig(save_path, dpi=300)
            QMessageBox.information(self, "Save Successful", f"The line chart has been saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"An error occurred while saving the line chart:\n{e}")

    def plot_msd_boxplot(self):
        """计算每组的gamma值并绘制箱型图。"""
        self.canvas.ax.clear()

        group_gamma_values = []  # 存储每组的gamma值
        group_labels = []        # 每组的标签

        # 获取全局时间间隔
        time_interval = self.time_interval_spinbox.value()

        if time_interval <= 0:
            QMessageBox.warning(self, "Invalid Input", "Time interval must be a positive integer.")
            return

        # 遍历每个组的输入
        for idx, input_line in enumerate(self.category_inputs):
            track_numbers = [int(x.strip()) for x in input_line.text().split(',') if x.strip().isdigit()]
            if not track_numbers:
                continue

            gamma_values = []  # 当前组的gamma值

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

                # 计算不同时间滞后的MSD
                for lag in range(1, total_frames):
                    displacements = (x_coords[lag:] - x_coords[:-lag]) ** 2 + (y_coords[lag:] - y_coords[:-lag]) ** 2
                    msd = displacements.mean()
                    msd = msd if msd > 0 else 1e-10  # 避免log(0)
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

                # 线性拟合
                slope, intercept, r_value, _, _ = linregress(log_time, log_msd)
                gamma_values.append(slope)

            if gamma_values:
                group_gamma_values.append(gamma_values)
                group_labels.append(f"Group {idx + 1}")

        if not group_gamma_values:
            QMessageBox.warning(self, "No Data", "No gamma values were calculated.")
            return

        # 绘制箱型图
        box = self.canvas.ax.boxplot(
            group_gamma_values, labels=group_labels, patch_artist=True,
            boxprops=dict(facecolor='#4D4D4D', color='#4D4D4D'),
            medianprops=dict(color='yellow'),
            whiskerprops=dict(color='#4D4D4D'),
            capprops=dict(color='#4D4D4D'),
            flierprops=dict(markerfacecolor='red', marker='o', markersize=5, linestyle='none')
        )

        # 设置箱型图的颜色
        for patch in box['boxes']:
            patch.set_facecolor('#4D4D4D')

        #self.canvas.ax.set_title("γ per Group", fontsize=16, pad=20, color='#2E2E2E')
        self.canvas.ax.set_xlabel("Groups", fontsize=14, labelpad=12, color='#4D4D4D')
        self.canvas.ax.set_ylabel("γ", fontsize=14, labelpad=12, color='#4D4D4D')
        self.canvas.ax.grid(True, linestyle=':', color='grey', alpha=0.7)

        self.canvas.draw()

        # 保存箱型图
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
        """查找并显示 cell_track_output_pictures 文件夹中编号最小的图片，支持鼠标交互"""
        image_folder = self.image_folder  # 使用传入的 image_folder
        if not os.path.exists(image_folder):
            QMessageBox.warning(self, 'Error', f'The directory {image_folder} does not exist.')
            return

        # 获取所有图片文件
        images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
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


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 确保你的config.py文件中定义了以下变量
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
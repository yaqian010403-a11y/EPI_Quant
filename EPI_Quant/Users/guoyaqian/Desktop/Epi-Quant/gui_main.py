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
from gui_cell_track import ComparisonMainWindow  # 导入追踪结果展示及纠错窗口
from gui_cell_direction import CellDirection  # 导入追踪结果展示及纠错窗口
from gui_plot_track import MainWindow as PlotTrackWindow  # 导入轨迹展示窗口
from gui_comparison_2 import MainWindow as CellClassification  # 导入细胞纠错窗口
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

        # 创建布局
        layout = QVBoxLayout(self)

        # 创建图片展示的布局（两行两列）
        grid_layout = QGridLayout()

        # 获取图片路径
        image_folder = config.cell_classification_output_path
        #print(image_folder)
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 只取前四张图片
        images_to_display = image_files[:]

        # 遍历图片，创建标签并添加到网格布局
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
        self.setGeometry(100, 100, 1000, 500)  # 调整宽度以容纳图片

        # 主布局设置为水平布局
        main_layout = QHBoxLayout()

        # 左侧：特征选择部分
        feature_layout = QVBoxLayout()

        # 添加排除边界细胞的复选框
        self.exclude_edge_cells_checkbox = QCheckBox("Exclude Edge Cells")
        feature_layout.addWidget(self.exclude_edge_cells_checkbox)  # 将复选框添加到左侧布局的顶部

        # 使用 QGroupBox 来将特征选择区域分组
        feature_group_box = QGroupBox("Select Features for Cell Classification")
        feature_group_layout = QGridLayout()

        self.instruction_label = QLabel()
        feature_layout.addWidget(self.instruction_label)

        self.checkboxes = []
        features = [
            'Cell Area', 'Fitted Ellipse Major Axis', 'Fitted Ellipse Minor Axis', 'Ellipse Major/Minor Axis Ratio', 'Fitted Ellipse Angle', 'Cell Perimeter', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio', 'Approximate Polygon Vertex Count',
            'Circumscribed Circle Radius', 'Inscribed Circle Radius', 'Circularity'
        ]
        # 设置复选框之间的间距
        feature_group_layout.setSpacing(15)  # 设置复选框之间的垂直和水平间距

        # 创建复选框并将其排列为两列
        for i, feature in enumerate(features):
            checkbox = QCheckBox(feature)
            row = i // 1
            col = i % 1
            feature_group_layout.addWidget(checkbox, row, col)
            self.checkboxes.append(checkbox)

        feature_group_box.setLayout(feature_group_layout)
        feature_layout.addWidget(feature_group_box)

        # 全选和取消全选的按钮
        select_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all)
        select_layout.addWidget(self.select_all_button)

        self.deselect_all_button = QPushButton(" Deselect All")
        self.deselect_all_button.clicked.connect(self.deselect_all)
        select_layout.addWidget(self.deselect_all_button)

        feature_layout.addLayout(select_layout)

        # 确认按钮
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.accept)
        feature_layout.addWidget(self.confirm_button)

        # 设置 feature_layout 的间距和边距，使其更加美观
        feature_layout.setSpacing(10)
        feature_layout.setContentsMargins(20, 20, 20, 20)

        # 创建一个新的水平布局以居中对齐
        centered_layout = QHBoxLayout()
        centered_layout.addLayout(feature_layout)
        centered_layout.setAlignment(Qt.AlignCenter)

        # 将居中对齐的布局添加到主布局
        main_layout.addLayout(centered_layout)

        # 右侧：显示图片
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(900, 600)  # 设置图片显示大小
        main_layout.addWidget(self.image_label)

        # 加载程序所在目录下的唯一图片
        self.load_image_from_folder()

        # 设置主布局
        self.setLayout(main_layout)

    def load_image_from_folder(self):
        """加载程序所在目录下的图片"""
        folder_path = os.path.join(os.path.dirname(__file__), 'images')  # 假设图片放在程序目录的 images 文件夹中
        if os.path.exists(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_path = os.path.join(folder_path, image_files[0])  # 假设只有一张图片
                pixmap = QPixmap(image_path)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=1,
                                              transformMode=Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("No images found")
        else:
            self.image_label.setText("Image folder does not exist")

    def select_all(self):
        """将所有复选框设为选中"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def deselect_all(self):
        """将所有复选框设为不选中"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)

    def get_selected_features(self):
        """获取所有选中的特征"""
        selected_features = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        return selected_features

    def exclude_edge_cells(self):
        """获取是否排除边界细胞的选项"""
        return self.exclude_edge_cells_checkbox.isChecked()  # 返回复选框的选中状态


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setStyleSheet("background-color:white;")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding around the plot

        # 初始化交互变量
        self.drag_start = None  # 记录拖动开始的位置
        self.zoom_factor = 0.9  # 缩放因子
        self.ax = None  # 当前活动的子图
        self.connect_events()

    def connect_events(self):
        """连接鼠标事件"""
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        """鼠标按下事件"""
        if event.inaxes:  # 鼠标在子图内按下
            self.ax = event.inaxes
            self.drag_start = (event.xdata, event.ydata)

    def on_motion(self, event):
        """鼠标拖动事件"""
        if self.drag_start is None or not event.inaxes or event.inaxes != self.ax:
            return
        dx = self.drag_start[0] - event.xdata
        dy = self.drag_start[1] - event.ydata

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        self.ax.set_xlim([x_min + dx, x_max + dx])
        self.ax.set_ylim([y_min + dy, y_max + dy])
        self.drag_start = (event.xdata, event.ydata)  # 更新起始点
        self.draw()

    def on_release(self, event):
        """鼠标释放事件"""
        self.drag_start = None

    def on_scroll(self, event):
        """鼠标滚轮事件（缩放）"""
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
        self.setFixedSize(1300, 950)  # 将窗口大小固定为 1200x850 像素

        main_layout = QVBoxLayout()

        # 调整整个主布局的边距，使路径输入区域更靠上
        main_layout.setContentsMargins(10, 0, 10, 10)  # 左、上、右、下的边距，将顶部边距设为5

        # 创建路径输入区域，并设置灰色背景
        path_groupbox = QGroupBox()  # 不再设置标题
        path_groupbox.setStyleSheet("""
                QGroupBox {
                    background-color: #DCDCDC;  /* 设置背景为灰色 */
                    border: 1px solid lightgray;  /* 设置边框为浅灰色 */
                    margin-top: 0px;  /* 减小上边距，设置为0 */
                }
            """)

        path_groupbox.setFixedHeight(120)  # 设置固定高度，限制占用空间

        path_layout = QVBoxLayout(path_groupbox)  # 将布局直接关联到 QGroupBox 上
        path_layout.setSpacing(0)

        # 统一设置按钮宽度
        button_width = 70  # 设置按钮宽度为 50 像素

        # 定义一个固定宽度，适用于所有输入框
        input_box_width = 1030  # 这里设置输入框的宽度

        # 原始图像路径
        img_path_hbox = QHBoxLayout()  # 创建另一个水平布局容器
        self.img_path_label = QLabel('Path to Original Images:')
        self.img_path_input = QLineEdit(self)
        self.img_path_input.setFixedWidth(input_box_width)  # 设置输入框的固定宽度
        self.img_browse_button = QPushButton("Browse")  # 创建浏览按钮
        self.img_browse_button.setFixedWidth(button_width)  # 设置按钮宽度
        self.img_browse_button.clicked.connect(self.browse_img_path)  # 绑定按钮点击事件

        img_path_hbox.addWidget(self.img_path_label)
        img_path_hbox.addStretch()  # 添加伸展项，将输入框推到右侧
        img_path_hbox.addWidget(self.img_path_input)  # 添加路径输入框
        img_path_hbox.addWidget(self.img_browse_button)  # 添加浏览按钮
        path_layout.addLayout(img_path_hbox)  # 将水平布局放入垂直布局

        # npy 文件路径
        npy_path_hbox = QHBoxLayout()  # 创建水平布局容器
        self.npy_path_label = QLabel('Path to .npy Files:')
        self.npy_path_input = QLineEdit(self)
        self.npy_path_input.setFixedWidth(input_box_width)  # 设置输入框的固定宽度
        self.npy_browse_button = QPushButton("Browse")  # 创建浏览按钮
        self.npy_browse_button.setFixedWidth(button_width)  # 设置按钮宽度
        self.npy_browse_button.clicked.connect(self.browse_npy_path)  # 绑定按钮点击事件

        npy_path_hbox.addWidget(self.npy_path_label)  # 添加标签
        npy_path_hbox.addStretch()  # 添加伸展项，将输入框推到右侧
        npy_path_hbox.addWidget(self.npy_path_input)  # 添加路径输入框
        npy_path_hbox.addWidget(self.npy_browse_button)  # 添加浏览按钮
        path_layout.addLayout(npy_path_hbox)  # 将水平布局放入垂直布局

        # 输出文件夹路径
        output_path_hbox = QHBoxLayout()  # 创建另一个水平布局容器
        self.output_path_label = QLabel('Output Folder Path:')
        self.output_path_input = QLineEdit(self)
        self.output_path_input.setFixedWidth(input_box_width)  # 设置输入框的固定宽度
        self.output_browse_button = QPushButton("Browse")  # 创建浏览按钮
        self.output_browse_button.setFixedWidth(button_width)  # 设置按钮宽度
        self.output_browse_button.clicked.connect(self.browse_output_path)  # 绑定按钮点击事件

        output_path_hbox.addWidget(self.output_path_label)
        output_path_hbox.addStretch()  # 添加伸展项，将输入框推到右侧
        output_path_hbox.addWidget(self.output_path_input)  # 添加路径输入框
        output_path_hbox.addWidget(self.output_browse_button)  # 添加浏览按钮
        path_layout.addLayout(output_path_hbox)  # 将水平布局放入垂直布局

        # 将路径输入区域添加到主布局中
        main_layout.addWidget(path_groupbox)

        # 绑定输入路径框的文本变化事件
        self.npy_path_input.textChanged.connect(self.update_paths_and_create_folders)
        self.img_path_input.textChanged.connect(self.update_paths_and_create_folders)
        self.output_path_input.textChanged.connect(self.update_paths_and_create_folders)
        # 创建图像显示和功能按钮区域
        content_layout = QHBoxLayout()

        # 左侧区域，包含按钮和图像显示区域
        left_layout = QVBoxLayout()

        # 使用 QTabWidget 代替三个按钮
        self.tab_widget = QTabWidget()

        # 设置 tab_widget 的宽度以填充布局
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tab_widget.setSizePolicy(size_policy)

        # 创建五个标签页
        self.original_image_tab = QWidget()
        self.classification_image_tab = QWidget()
        self.tracking_image_tab = QWidget()

        # 为 QTabWidget 和 QTabBar 设置样式表，隐藏灰色背景条
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {  /* 隐藏标签页下方的背景 */
                border: 0px;
                background: transparent;
            }
            QTabBar::tab {
                background-color: white;  /* 设置标签的背景颜色 */
                border: 1px solid lightgray;  /* 设置标签的边框 */
                padding: 3px;
                min-width: 315px;  /* 设置标签的最小宽度为 250 像素 */
                border-radius: 0px;  /* 设置边框的弧度为 10 像素 */
            }
            QTabBar::tab:selected {  /* 设置选中状态的标签样式 */
                background-color: blue;  /* 让整个标签背景变为蓝色 */
                color: white;  /* 选中时文本颜色为白色，确保可见 */
                border: 2px solid blue;  /* 设置整个标签的边框颜色为蓝色 */
            }
            QTabBar::tab:disabled {  /* 禁用状态下的标签样式 */
                background-color: #D3D3D3;  /* 禁用时背景为灰色 */
                color: darkgray;  /* 禁用时文本颜色为深灰色 */
                border: 1px solid  #D3D3D3;  /* 禁用时边框为灰色 */
    }
        """)

        # 添加标签页到 QTabWidget
        self.tab_widget.addTab(self.original_image_tab, "Original Cell Images")
        self.tab_widget.addTab(self.classification_image_tab, "Cell Classification Images")
        self.tab_widget.addTab(self.tracking_image_tab, "Cell Tracking Images")

        # 禁用标签页
        self.tab_widget.setTabEnabled(0, False)  # 禁用"原始细胞图像"标签页
        self.tab_widget.setTabEnabled(1, False)  # 禁用"细胞分类图像"标签页
        self.tab_widget.setTabEnabled(2, False)  # 禁用"细胞追踪图像"标签页

        # 连接标签切换信号到图像加载函数
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # 添加标签到布局，去除间距和填充
        left_layout.addWidget(self.tab_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)  # 去除边距
        left_layout.setSpacing(5)  # 减少组件之间的间距

        # 图像显示区域
        self.image_layout = QVBoxLayout()
        self.canvas_widget = QWidget()
        self.canvas_widget.setLayout(self.image_layout)

        self.canvas_widget.setStyleSheet("""
            background-color: #DCDCDC;  /* 设置为灰白色 */
        """)

        # 将 canvas_widget 放在靠上的位置
        left_layout.addWidget(self.canvas_widget)

        # 添加垂直滚动条
        self.scroll_bar = QScrollBar(Qt.Vertical)
        self.scroll_bar.setMinimum(1)  # 滚动条最小值
        self.scroll_bar.valueChanged.connect(self.scroll_value_changed)  # 连接滚动条改变信号
        self.scroll_bar.setEnabled(False)  # 初始禁用，待加载图片后启用

        # 将图片展示区域和滚动条布局到内容区域
        content_layout.addLayout(left_layout, 3)
        content_layout.addWidget(self.scroll_bar, 1)  # 右侧放置滚动条

        # 创建 feature_layout 用于右侧功能按钮
        feature_layout = QVBoxLayout()

        # 定义按钮样式为类的属性
        self.button_style_disabled = """
                QPushButton {
                    background-color: #D3D3D3;   /* 灰白色背景 */
                    color: darkgray;              /* 文本颜色 */
                    border: 1px solid gray;       /* 边框颜色和厚度 */
                    border-radius: 5px;           /* 设置为0，确保没有圆角 */
                    font-size: 14px;              /* 设置文本字体大小 */
                }
            """

        self.button_style_enabled = """
                QPushButton {
                    background-color: white;      /* 背景颜色 */
                    color: black;                 /* 文本颜色 */
                    border: 1px solid black;      /* 边框颜色和厚度 */
                    border-radius: 5px;           /* 设置为0，确保没有圆角 */
                    font-size: 14px;              /* 设置文本字体大小 */
                }
            """

        # **工作区按钮**
        work_area_group = QGroupBox("Work Area")
        work_area_layout = QVBoxLayout()
        work_area_group.setFixedHeight(300)  # 将高度调整为 300 像素

        # 统一按钮间距和边距
        work_area_layout.setSpacing(10)
        work_area_layout.setContentsMargins(10, 10, 10, 10)  # 左、上、右、下的边距

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

        # 将工作区布局添加到 QGroupBox 中
        work_area_group.setLayout(work_area_layout)

        # **展示区按钮**
        display_area_group = QGroupBox("Display Area")
        display_area_layout = QVBoxLayout()
        display_area_group.setFixedHeight(300)  # 将高度调整为 300 像素

        # 统一按钮间距和边距
        display_area_layout.setSpacing(10)
        display_area_layout.setContentsMargins(10, 10, 10, 10)  # 左、上、右、下的边距

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

        # 将展示区布局添加到 QGroupBox 中
        display_area_group.setLayout(display_area_layout)

        # **执行操作记录区**
        operations_log_group = QGroupBox("Operation Log")
        operations_log_layout = QVBoxLayout()
        operations_log_group.setFixedHeight(100)  # 调整高度为 100 像素

        # 使用 QTextEdit 操作日志
        self.operations_log = QTextEdit()
        self.operations_log.setReadOnly(True)  # 设置为只读
        self.operations_log.setEnabled(False)  # 禁用操作日志

        operations_log_layout.addWidget(self.operations_log)
        operations_log_group.setLayout(operations_log_layout)

        # **将工作区、展示区和操作记录区添加到 feature_layout 中**
        feature_layout.addWidget(work_area_group)  # 添加工作区
        feature_layout.addWidget(display_area_group)  # 添加展示区

        # 将布局设置到窗口
        content_layout.addLayout(feature_layout, 1)

        self.canvas_widget.setFixedHeight(self.canvas_widget.height() + 185)
        # work_area_group.setFixedHeight(work_area_group.height() + 100)

        main_layout.addLayout(content_layout)

        main_layout.addWidget(operations_log_group)  # 在底部添加操作日志区域

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 初始化后禁用所有按钮
        # self.disable_buttons()

    def update_paths_and_create_folders(self):
        # 获取并更新路径
        npy_path = self.npy_path_input.text().strip()
        img_path = self.img_path_input.text().strip()
        output_path = self.output_path_input.text().strip()

        # 更新 config 对象中的路径
        config.npy_folder_path = npy_path
        config.Img_path = img_path
        config.output_path = output_path
        config.cell_classification_output_path = os.path.join(output_path, 'cell_classification_output')
        config.cell_track_output_path = os.path.join(output_path, 'cell_track_output')
        config.quantitative_analysis_output_path = os.path.join(output_path, 'quantitative_analysis_output')

        # 检查路径是否全部输入
        if not (npy_path and img_path and output_path):
            self.disable_buttons()  # 如果路径未全部填写，则禁用按钮
            self.tab_widget.setEnabled(False)
            return

        try:
            # 尝试创建输出目录和子目录
            if not os.path.exists(output_path):
                #print(f"尝试创建目录: {output_path}")
                os.makedirs(output_path)

            if not os.path.exists(config.cell_classification_output_path):
                #print(f"尝试创建目录: {config.cell_classification_output_path}")
                os.makedirs(config.cell_classification_output_path)

            if not os.path.exists(config.cell_track_output_path):
                #print(f"尝试创建目录: {config.cell_track_output_path}")
                os.makedirs(config.cell_track_output_path)

            if not os.path.exists(config.quantitative_analysis_output_path):
                #print(f"尝试创建目录: {config.quantitative_analysis_output_path}")
                os.makedirs(config.quantitative_analysis_output_path)

            self.tab_widget.setTabEnabled(0, True)  # 启用"原始细胞图像"标签页
            self.tab_widget.setTabEnabled(1, True)  # 启用"细胞分类图像"标签页
            self.tab_widget.setTabEnabled(2, True)  # 启用"细胞追踪图像"标签页

            # 启用标签和功能按钮
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

            # 主动加载原始细胞图像
            self.load_images(config.Img_path, image_type="original")

            # 检查并禁用"细胞分类图像"和"细胞追踪图像"标签
            self.check_folder_and_toggle_tab(
                os.path.join(config.cell_classification_output_path, 'cells_clustering_results_pictures'), 1)
            self.check_folder_and_toggle_tab(os.path.join(config.cell_track_output_path, 'cell_track_output_pictures'),
                                             2)

            # 确保初始化时默认选中第一个标签页（原始细胞图像）
            self.tab_widget.setCurrentIndex(0)

            # 启用操作日志
            self.operations_log.setEnabled(True)


        except OSError as e:
            # 捕获文件夹创建错误并显示详细的错误信息
            QMessageBox.warning(self, "Error", f"Folder creation failed: {e.strerror}")

    def log_operation(self, operation_text, start_time=None):
        """
        记录并显示已执行的操作，包含开始和结束时间。
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if start_time:
            # 记录操作结束时间
            operation_message = f"Completed: {operation_text} at {current_time}"
        else:
            # 记录操作开始时间
            operation_message = f"Started: {operation_text} at {current_time}"

        self.operations_log.append(operation_message)
        self.operations_log.repaint()  # 强制刷新操作日志的显示

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
        if os.path.exists(folder_path) and os.listdir(folder_path):  # 确保文件夹存在且非空
            self.tab_widget.setTabEnabled(tab_index, True)
        else:
            self.tab_widget.setTabEnabled(tab_index, False)

    def on_tab_changed(self, index):
        original_cells_path = self.img_path_input.text()

        # 获取最新的细胞分类图像路径
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
            self.progress_dialog.close()  # 关闭进度条

            QMessageBox.warning(self, "Notice", "Cells_info.xlsx not found. Please extract cell features first.")
            return

        cells_info = pd.read_excel(cells_info_path)

        # 检查是否包含 'leading_edge' 列，以确定是否有两种细胞类型
        has_leading_edge = 'leading_edge' in cells_info.columns
        num_cell_types = 2 if has_leading_edge else 1

        if has_leading_edge:
            self.show_progress_bar("Starting classified cell tracking...")

            # 将细胞分为大细胞和小细胞
            big_cells_info = cells_info[cells_info['leading_edge'] < 0].copy()
            big_cells_info_path = os.path.join(config.cell_track_output_path, 'mes_cells_info.xlsx')
            big_cells_info.to_excel(big_cells_info_path, sheet_name='mesCells', index=False)

            small_cells_info = cells_info[cells_info['leading_edge'] > 0].copy()
            small_cells_info_path = os.path.join(config.cell_track_output_path, 'epi_cells_info.xlsx')
            small_cells_info.to_excel(small_cells_info_path, sheet_name='epiCells', index=False)

            big_cells_info.set_index('Cell Index', inplace=True)
            small_cells_info.set_index('Cell Index', inplace=True)

            all_tracking_data = pd.DataFrame()

            # 定义进度条分段
            # 假设细胞跟踪部分占总进度的 50%，大细胞和小细胞各占 25%
            tracking_start = 5
            tracking_end = 50
            half_progress = (tracking_end - tracking_start) / 2  # 每种细胞类型分配的进度量（22.5%）

            # 处理大细胞
            if not big_cells_info.empty:
                big_tracking_data = self.run_cell_tracking(
                    big_cells_info,
                    progress_start=tracking_start,
                    progress_end=tracking_start + half_progress
                )
                all_tracking_data = pd.concat([all_tracking_data, big_tracking_data], axis=0)

            # 处理小细胞
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

        # 保存所有跟踪数据
        if all_tracking_data is not None and not all_tracking_data.empty:
            all_cells_tracking_output_folder = os.path.join(output_path, 'all_cell_tracking')
            all_cells_tracking_output_path = os.path.join(all_cells_tracking_output_folder,
                                                          'all_cell_merged_tracking_results.xlsx')
            os.makedirs(all_cells_tracking_output_folder, exist_ok=True)
            all_tracking_data.to_excel(all_cells_tracking_output_path, index=False)

        # 更新进度条到 50%（细胞跟踪部分完成）
        if has_leading_edge:
            self.progress_dialog.setValue(tracking_end)  # 50%
        else:
            self.progress_dialog.setValue(50)

        # 生成对比图像
        if all_cells_tracking_output_path:
            all_comparison_window = ComparisonMainWindow(
                npy_folder,
                img_folder,
                all_cells_tracking_output_path,
                'all_cells'
            )
            # 图像生成部分占剩余的进度（50% 到 100%）
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

                # 提示提取完成
                QMessageBox.information(self, "Completed", "Cell feature extraction has been completed.")

                return True

                # 启用其他功能按钮
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
        """判断某个细胞是否位于图像边缘或接近图像边缘一定距离内。"""
        min_x = np.min(cell_pixels[1])
        max_x = np.max(cell_pixels[1])
        min_y = np.min(cell_pixels[0])
        max_y = np.max(cell_pixels[0])

        # 判断细胞轮廓是否位于图像边界的 margin 范围内
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
        # 获取所有要处理的文件
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
            outlines = dat['outlines']  # 提取 outlines 数据
            height, width = outlines.shape  # 获取图像的高度和宽度
            fig_num += 1
            cells_info_fig = pd.DataFrame(columns=cells_info_columns)  # 构建空的 cells_info 存放信息
            regex = re.compile(r'\d+')
            fig_id = str(max(map(int, regex.findall(filename))))  # 计算图像编号
            frame_number = int(fig_id) if fig_id else idx + 1

            self.show_progress_bar(f'Extracting cell features for frame {frame_number}...')
            progress_value = int((idx + 1) / total_files * 100)  # 计算进度
            self.progress_dialog.setValue(progress_value)  # 更新进度条数值
            QApplication.processEvents()  # 强制刷新事件循环

            unique_ids = np.unique(outlines)  # 提取所有唯一的 ID
            unique_ids = unique_ids[unique_ids != 0]  # 排除背景 ID 0

            for cell_id in unique_ids:
                positions = np.where(outlines == cell_id)

                # 根据传入的参数，决定是否排除边界细胞
                if exclude_edge_cells:  # 如果选择了排除边界细胞
                    if self.is_edge_cell(positions, height, width, margin=3):
                        continue  # 如果是边界细胞，跳过

                contour_points = np.array([positions[1], positions[0]]).T.reshape(-1, 1, 2)

                # === 使用轮廓点进行面积、周长、多边形拟合 ===
                fill_img = np.zeros_like(outlines, dtype=np.uint8)
                cv2.fillPoly(fill_img, [contour_points], 1)
                cell_area = np.sum(fill_img)

                # 提取轮廓
                contours, _ = cv2.findContours(fill_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    filled_contour = contours[0]
                    cell_arclength = int(cv2.arcLength(filled_contour, True))
                    vertices = cv2.approxPolyDP(filled_contour, 2, True)

                    # 计算最小外接圆和最大内接圆
                    _, radius = cv2.minEnclosingCircle(filled_contour)
                    min_enclosing_circle_radius = int(radius)

                    dist_transform = cv2.distanceTransform(fill_img, cv2.DIST_L2, 5)
                    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
                    max_inner_circle_radius = int(max_val)

                    # 检查 cell_arclength 是否为 0，避免除以 0
                    if cell_arclength > 0:
                        circularity = (4 * math.pi * cell_area) / (cell_arclength ** 2)
                        p_value = (4 * math.pi) / (circularity ** 2)
                    else:
                        circularity = -1  # 或者设置为其他合理的值
                        p_value = -1
                else:
                    vertices = []
                    min_enclosing_circle_radius = 0
                    max_inner_circle_radius = 0

                # === 椭圆拟合 ===
                if len(contour_points) >= 5:
                    cell_ellipse = cv2.fitEllipse(contour_points)
                    ratio_long_and_short_axes = cell_ellipse[1][1] / cell_ellipse[1][0] if cell_ellipse[1][
                                                                                               0] != 0 else -1
                    cell_ellipse_angle = 180 - cell_ellipse[2] if cell_ellipse[2] > 90 else cell_ellipse[2]
                else:
                    ratio_long_and_short_axes = -1
                    cell_ellipse_angle = -1

                # 计算中心点 (cx, cy)
                cy = int(np.mean(positions[0]))
                cx = int(np.mean(positions[1]))

                # 计算 AP/DV 的值
                ap = np.max(positions[1]) - np.min(positions[1])
                dv = np.max(positions[0]) - np.min(positions[0])
                aspect_ratio = ap / dv if dv != 0 else -1

                # 保存信息
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

        # 将包含-1的行过滤掉
        # cells_info = cells_info[~(cells_info == -1).any(axis=1)]
        print(cells_info[(cells_info == -1).any(axis=1)])
        # 将"细胞编号"作为第一列导出到Excel文件
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

        # 记录开始时间
        start_time = time.time()

        npy_folder = config.npy_folder_path

        # 确保路径存在
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



        # 直接打开纠错窗口并传递文件路径和细胞类型（不再传递帧号）
        self.open_tracking_correction_window(tracking_data_path, cell_type)

        # 检查文件夹并更新界面
        self.check_folder_and_toggle_tab(config.cell_track_output_path, 2)

        self.log_operation("Correct Cell Tracking", start_time=start_time)

    def open_tracking_correction_window(self, tracking_data_path, cell_type):
        npy_folder = config.npy_folder_path  # 从配置中获取 npy 文件夹路径
        img_folder = config.Img_path

        # 实例化 ComparisonMainWindow，并传递 cell_type
        self.comparison_window = ComparisonMainWindow(npy_folder, img_folder,  tracking_data_path, cell_type)

        # 直接显示纠错窗口
        self.comparison_window.show()

    def show_correction_window(self):
        # 记录开始时间
        start_time = time.time()

        # 记录开始执行细胞分类
        self.log_operation("Correct Cell Classification", start_time=None)

        clustering_result_path = os.path.join(config.cell_classification_output_path, 'cells_clustering_results.xlsx')
        cells_info_path = os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx')
        if not os.path.exists(clustering_result_path) or not os.path.exists(cells_info_path):
            QMessageBox.warning(self, "Notice", "Cell classification results not found. Please perform cell classification first.")
            return


        # 通过顶部帧选择输入框获取帧号
        #frame_number = int(self.frame_input.text())  # 假设 frame_input 是顶部输入框的对象
        #print(f"选择的帧号: {frame_number}")

        # 打开 CellClassification 窗口并跳转到指定的帧
        self.correction_window = CellClassification(
            npy_folder=config.npy_folder_path,
            img_folder=config.Img_path,
            output_picture=os.path.join(config.cell_classification_output_path,
                                        'cells_clustering_results_pictures'),
            clustering_data_path=os.path.join(config.cell_classification_output_path,
                                              'cells_clustering_results.xlsx'),
            cells_info_path=os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx'),
            output_folder=config.cell_classification_output_path,
            #start_frame=frame_number,  # 传递帧号给 CellClassification
        )
        self.correction_window.show()

        # 使用已有函数启用"细胞分类图像"标签
        self.check_folder_and_toggle_tab(config.cell_classification_output_path, 1)

        # 记录完成操作的时间
        self.log_operation("Correct Cell Classification", start_time=start_time)

    def load_images(self, folder_path, image_type="other"):
        """加载图像并在滚动区域中展示"""
        # 清空图像布局中的所有图像
        for i in reversed(range(self.image_layout.count())):
            widget_to_remove = self.image_layout.takeAt(i).widget()  # 使用 takeAt() 移除
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()  # 删除组件

        # 保存当前文件夹路径，确保 display_image 方法可以使用
        self.current_folder_path = folder_path  # 这里赋值

        # 加载图像文件
        self.image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda f: int(re.search(r'\d+', f).group()) if re.search(r'\d+', f) else float('inf')
        )

        if not self.image_files:
            return  # 如果没有图片，直接返回

        # 计算滚动条的最大值，根据图片数量和每页显示的图片数
        max_value = max(1, (len(self.image_files) + 3) // 4)  # 每页显示4张图片
        self.scroll_bar.setMaximum(max_value)  # 设置滚动条最大值为页面数量
        self.scroll_bar.setEnabled(True)  # 启用滚动条

        # 将滚动条重置为最上方（最小值）
        self.scroll_bar.setValue(1)

        # 加载第一组图片
        self.display_images(0)

    def display_images(self, start_index=0):
        """显示从 start_index 开始的 2x2 图片，并在每张图片下方显示文件名"""
        total_images = len(self.image_files)
        num_images = min(max(total_images - start_index, 0), 4)  # 确保 num_images 不小于0且不超过4

        if num_images == 0:
            #print("没有可显示的图片。")
            return  # 没有图片，直接返回

        # 清空当前布局中的图像
        for i in reversed(range(self.image_layout.count())):
            widget_to_remove = self.image_layout.takeAt(i).widget()  # 使用 takeAt() 移除
            if widget_to_remove is not None:
                widget_to_remove.deleteLater()  # 删除组件

        # 创建图像展示的 canvas
        canvas = MplCanvas(self, width=8.00, height=8.00, dpi=100)

        # 使用 subplots 创建 2x2 的布局
        axes = canvas.fig.subplots(2, 2, gridspec_kw={'wspace': 0.2, 'hspace': 0.2})  # 增加行列间距
        axes = axes.flatten()  # 将 2x2 的二维数组展平成一维

        # 遍历子图并加载图片
        for idx, ax in enumerate(axes):
            if idx < num_images:
                image_file = self.image_files[start_index + idx]
                image_path = os.path.join(self.current_folder_path, image_file)


                try:
                    # 加载图片
                    img_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if img_data is not None:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

                        # 显示图片
                        ax.imshow(img_data, aspect='auto')  # 保持图片比例
                        ax.axis('off')  # 隐藏坐标轴

                        # 在图片下方显示文件名
                        ax.set_title(image_file, fontsize=10, pad=5, loc='center')
                    else:
                        raise ValueError("Image Load Error")
                except Exception as e:
                    #print(f"错误加载图像 {image_path}: {e}")
                    # 如果图片未加载成功，显示警告文本
                    ax.text(0.5, 0.5, 'Failed to Load Image', ha='center', va='center',
                        fontsize=12, color='red', transform=ax.transAxes
                    )
                    ax.axis('off')
            else:
                # 未分配图片的子图，隐藏
                ax.axis('off')

        # 调整画布边距，避免图片被压缩
        canvas.fig.subplots_adjust(wspace=0.1, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)
        # 将 canvas 添加到布局中
        self.image_layout.addWidget(canvas)
        canvas.draw_idle()  # 刷新画布

    def scroll_value_changed(self):
        """当滚动条的值改变时，更新显示的图片"""
        value = (self.scroll_bar.value() - 1) * 4  # 每次显示4张图

        # 保护性检查，确保值在合理范围内
        if value < 0:
            value = 0
        elif value >= len(self.image_files):
            value = len(self.image_files) - 4  # 显示最后一页图片，最多4张

        # 调用 display_images 来显示从 value 开始的图片
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
        # 提取出主成分分析得到的数据
        fac_index = cells_info_pca.index


        # 自动选择所有以 'PC' 开头的列作为主成分
        components = [col for col in cells_info_pca.columns[:-1] if col.startswith('PC')]

        # 提取所有主成分的数据
        data = cells_info_pca[components].values

        # 使用 K-Means 聚类，设置为两个聚类
        kmeans = KMeans(n_clusters=2, n_init=10)
        label = kmeans.fit_predict(data)

        # 绘制前两个主成分的散点图
        plt.figure(figsize=(10, 8), dpi=300)  # 设置高分辨率
        plt.scatter(data[:, 0], data[:, 1], c=label, cmap='viridis', s=5)
        plt.xlabel('PC1', fontsize=18, fontweight='bold')
        plt.ylabel('PC2', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')



        # 显示右边框和上边框的轴线
        ax = plt.gca()
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)

        # 设置坐标轴范围，留出一些边距，确保右边框显示
        ax.set_xlim(data[:, 0].min() - 1, data[:, 0].max() + 1)
        ax.set_ylim(data[:, 1].min() - 1, data[:, 1].max() + 1)

        # 使用逻辑回归拟合分割边界
        log_reg = LogisticRegression()
        log_reg.fit(data[:, :2], label)  # 只使用前两个主成分

        # 创建网格以绘制二维决策边界
        x_values = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
        y_values = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_values, y_values)
        grid = np.c_[xx.ravel(), yy.ravel()]
        probabilities = log_reg.predict_proba(grid)[:, 1].reshape(xx.shape)

        # 绘制决策边界
        plt.contour(xx, yy, probabilities, levels=[0.5], linestyles=['--'], colors='r')

        # 使用 tight_layout 并适当增加 pad
        plt.tight_layout(pad=1)

        # 保存图片
        plt.savefig(os.path.join(path, 'cells_PCA_2D.png'), dpi=300)
        plt.close()

        # 输出聚类结果
        if np.sum(label == 1) > np.sum(label == 0):
            small_cells_index = np.where(label == 1)
            big_cells_index = np.where(label == 0)
            label = np.where(label == 0, 1, 0)  # 反转标签
        else:
            big_cells_index = np.where(label == 1)
            small_cells_index = np.where(label == 0)
            label = np.where(label == 1, 1, 0)  # 保持标签

        # 生成预测结果
        big_cells_predict = list(fac_index[big_cells_index])
        small_cells_predict = list(fac_index[small_cells_index])

        # 将分类结果生成一个Excel表
        results = pd.DataFrame({
            'Cell_Index': fac_index,
            'Cluster_Label': label,
            'Cell_Type': ['mes cell' if lbl == 1 else 'epi cell' for lbl in label]
        })
        print(results)

        # 保存结果到Excel
        results.to_excel(os.path.join(path, 'cells_clustering_results.xlsx'), index=False)



        return big_cells_predict, small_cells_predict

    def show_plot_track_window(self):
        analysis_data_dir = os.path.join(config.quantitative_analysis_output_path,
                                         'all_cell_quantitative_analysis_output')
        if not os.path.exists(analysis_data_dir) or not os.listdir(analysis_data_dir):
            QMessageBox.warning(self, "Notice", "Quantitative analysis data not found. Please generate it first.")
            return
        # 记录开始时间
        start_time = time.time()

        # 记录开始执行细胞分类
        self.log_operation("Display Cell Tracking Trajectories", start_time=None)

        img_folder = config.Img_path
        npy_folder = config.npy_folder_path


        self.plot_track_window = PlotTrackWindow(img_folder,npy_folder)
        self.plot_track_window.show()
        self.log_operation("Display Cell Tracking Trajectories", start_time=start_time)

    def run_cell_tracking(self, cells_info, progress_start=0, progress_end=100):
        """
        处理细胞跟踪，并在指定的进度范围内更新进度条。

        :param cells_info: 需要跟踪的细胞信息 DataFrame。
        :param progress_start: 进度条更新的起始百分比。
        :param progress_end: 进度条更新的结束百分比。
        :return: 合并后的跟踪数据的 DataFrame。
        """
        # 确保索引都是字符串格式
        cells_info.index = cells_info.index.map(str)

        # 获取 Cells_info 的内容
        path = config.cell_track_output_path

        # 初始化 start_index
        start_index = 1
        merge_only = False

        # 判断 cells_info 中的 leading_edge 列的情况
        if 'leading_edge' not in cells_info.columns:
            # 如果没有 leading_edge 列
            folder_prefix = "All_cell_"
            output_subfolder = "All_cell_tracking/"
            name = 'all_cell'
        elif (cells_info['leading_edge'] <= 0).all():
            folder_prefix = "mes_cell_"
            output_subfolder = "mes_cell_tracking/"
            name = 'mes_cell'

            # 检查是否存在表皮细胞的合并文件
            small_cell_file = os.path.join(path, "epi_cell_tracking",
                                           "epi_cell_merged_tracking_results.xlsx")
            if os.path.exists(small_cell_file):
                small_cell_df = pd.read_excel(small_cell_file, header=0)
                if not small_cell_df.empty:
                    start_index = small_cell_df.iloc[:, 0].max() + 1
            # else:
            # 未找到表皮细胞的合并文件，序号从1开始

        elif (cells_info['leading_edge'] > 0).all():
            folder_prefix = "epi_cell_"
            output_subfolder = "epi_cell_tracking/"
            name = 'epi_cell'

            # 检查是否存在羊浆膜细胞的合并文件
            big_cell_file = os.path.join(path, "mes_cell_tracking", "mes_cell_merged_tracking_results.xlsx")
            if os.path.exists(big_cell_file):
                big_cell_df = pd.read_excel(big_cell_file, header=0)
                if not big_cell_df.empty:
                    start_index = big_cell_df.iloc[:, 0].max() + 1
            # else:
            # 未找到羊浆膜细胞的合并文件，序号从1开始

        output_path = os.path.join(path, output_subfolder)
        os.makedirs(output_path, exist_ok=True)

        # 设置 pandas 显示选项（可选，影响调试输出）
        pd.set_option('display.max_row', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.unicode.east_asian_width', True)
        pd.set_option('display.width', 180)

        # 追踪和匹配逻辑

        # 提取所有独特的帧编号
        fig_ids = sorted(set(
            int(re.search(r'Cell(\d+)_', idx).group(1)) for idx in cells_info.index if re.search(r'Cell(\d+)_', idx)))
        total_figs = len(fig_ids)
        if total_figs == 0:
            QMessageBox.warning(self, 'Warning', 'No valid frames found for tracking.')
            return pd.DataFrame()

        for idx, fig_id in enumerate(fig_ids):
            # 计算当前进度并更新进度条
            if total_figs > 0:
                local_progress = (idx + 1) / total_figs  # 0 到 1
                global_progress = progress_start + local_progress * (progress_end - progress_start)
                progress_value = min(int(global_progress), progress_end)
            else:
                progress_value = progress_end

            self.show_progress_bar(f"Tracking frame {fig_id} for {name}...")
            self.progress_dialog.setValue(progress_value)
            QApplication.processEvents()  # 强制刷新事件循环

            # 获取当前帧和下一帧的细胞信息
            cells_info_fig1 = cells_info[cells_info.index.str.contains(f'Cell{fig_id}_')]
            cells_info_fig2 = cells_info[cells_info.index.str.contains(f'Cell{fig_id + 1}_')]
            cells_info_fig1_index = list(cells_info_fig1.index)
            cells_info_fig2_index = list(cells_info_fig2.index)

            # 计算细胞之间的关系
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

                    # 计算欧氏距离
                    cell1_cell2_distance = 0
                    for cell_nature_id in range(len(cells_nature)):
                        cell1_cell2_distance += cells_weight[cell_nature_id] * (
                                cell2_info[cell_nature_id] - cell1_info[cell_nature_id]) ** 2
                    cell1_cell2_distance = np.sqrt(cell1_cell2_distance)

                    cells_relation[cell1_id, cell2_id] = cell1_cell2_distance

            # 保存 cells_relation 到 Excel 文件
            relation_df = pd.DataFrame(cells_relation, index=cells_info_fig1_index, columns=cells_info_fig2_index)
            relation_file_path = os.path.join(output_path, f'{folder_prefix}Cell_Relation_{fig_id}_{fig_id + 1}.xlsx')
            relation_df.to_excel(relation_file_path)

            df = pd.DataFrame(cells_relation, index=cells_info_fig1_index, columns=cells_info_fig2_index)

            max_euclidean_distance = cells_info['Circumscribed Circle Radius'].max()

            # 使用匈牙利算法进行匹配
            df_filled = df.fillna(max_euclidean_distance)
            row_ind, col_ind = linear_sum_assignment(df_filled.values)

            matching_result = pd.DataFrame({
                f'Frame{fig_id}': [df.index[r] if r < len(df.index) else f"Cell{fig_id}_" for r in row_ind],
                f'Frame{fig_id + 1}': [df.columns[c] if c < len(df.columns) else f"Cell{fig_id + 1}_" for c in col_ind],
                'Difference': [df_filled.iat[r, c] for r, c in zip(row_ind, col_ind)]
            })

            # 将匹配结果的序号从 start_index 开始编号
            matching_result.insert(0, 'Index', range(start_index, start_index + len(matching_result)))

            # 过滤掉差异度值等于 max_euclidean_distance 的行
            matching_result = matching_result[matching_result['Difference'] != max_euclidean_distance]

            # 保存匹配结果
            filtered_output_file_path = os.path.join(output_path,
                                                     f'{folder_prefix}Matching_Results_new_{fig_id}_{fig_id + 1}.xlsx')
            matching_result.to_excel(filtered_output_file_path, index=False)

            # 更新 start_index 为下一个编号的起点
            start_index += len(matching_result)

        # 合并追踪结果
        matching_files = [f for f in os.listdir(output_path) if
                          re.match(rf'{folder_prefix}Matching_Results_new_\d+_\d+\.xlsx', f)]
        matching_frames = sorted(
            set(int(re.search(r'\d+', f).group(0)) for f in matching_files if re.search(r'\d+', f)))
        start_frame = min(matching_frames) if matching_frames else 1
        files_count = len(matching_frames)


        # 初始化，加载第一个文件
        if matching_files:
            first_file_path = os.path.join(output_path,
                                           f'{folder_prefix}Matching_Results_new_{start_frame}_{start_frame + 1}.xlsx')
            data_final = self.load_and_process_data(first_file_path)
            data_final.columns = ['Index', f'Frame{start_frame}', f'Frame{start_frame + 1}']

            # 构造需要添加的新列名
            all_new_cols = [f'Frame{i + 1}' for i in range(start_frame + 1, start_frame + files_count - 1)]

            # 用 pd.concat 一次性添加列，避免碎片化
            extra_columns_df = pd.DataFrame({col: [None] * len(data_final) for col in all_new_cols})
            data_final = pd.concat([data_final, extra_columns_df], axis=1)

            # 处理后续文件
            for i in range(start_frame + 1, start_frame + files_count - 1):
                next_file_path = os.path.join(output_path, f'{folder_prefix}Matching_Results_new_{i}_{i + 1}.xlsx')
                data_next = self.load_and_process_data(next_file_path)
                data_next.columns = ['Index', f'Frame{i}', f'Frame{i + 1}']

                for index, row in data_next.iterrows():
                    match_index = data_final[data_final[f'Frame{i}'] == row[f'Frame{i}']].index
                    if not match_index.empty:
                        data_final.at[match_index[0], f'Frame{i + 1}'] = row[f'Frame{i + 1}']

            # 最后一次 copy（不一定必要了）
            # data_final = data_final.copy()

        else:
            data_final = pd.DataFrame()

        # 保存合并后的结果
        merged_output_path = os.path.join(output_path, f'{folder_prefix}merged_tracking_results.xlsx')
        data_final.to_excel(merged_output_path, index=False)

        # 返回合并后的数据
        return data_final

    def load_and_process_data(self, file_path):
        # 加载Excel文件
        data = pd.read_excel(file_path)
        # 删除不需要的"差异度"列
        data.drop(columns=['Difference'], inplace=True)
        return data

    def generate_and_notify(self):

        # 记录开始时间
        start_time = time.time()


        # 创建并显示初始进度条
        self.show_progress_bar("Generating quantitative analysis data, please wait...")
        # 设置进度条初始值
        self.progress_dialog.setValue(1)

        tracking_data_path = os.path.join(config.cell_track_output_path, 'all_cell_tracking', 'all_cell_merged_tracking_results.xlsx')
        if not os.path.exists(tracking_data_path):
            self.progress_dialog.close()  # ✅ 加这一句关闭进度条
            QMessageBox.warning(self, "Notice", "Cell tracking results not found. Please perform cell tracking first.")
            return

        # 记录开始执行细胞分类
        self.log_operation("Generating quantitative analysis data", start_time=None)

        all_cells_output_dir = self.generate_quantitative_analysis_data('all_cell_tracking',
             'all_cell_merged_tracking_results_updated.xlsx',
            'all_cell_merged_tracking_results.xlsx', 'all_cell_quantitative_analysis_output'
        )
        self.progress_dialog.setValue(100)
        self.progress_dialog.close()

        # 通知用户所有数据已生成
        QMessageBox.information(
            self, "Data Generation Complete",
            f"Quantitative analysis data has been successfully generated and saved in the following directory:\n"
            f"All Cells:{all_cells_output_dir}\n"
        )
        self.log_operation("Generating quantitative analysis data", start_time=start_time)


    def generate_quantitative_analysis_data(self, tracking_folder, updated_filename, default_filename,
                                            output_folder_name):
        # 检查并选择要加载的 Excel 文件路径
        updated_file1_path = os.path.join(config.cell_track_output_path, tracking_folder, updated_filename)
        default_file1_path = os.path.join(config.cell_track_output_path, tracking_folder, default_filename)

        # 如果更新后的文件存在，则使用它；否则使用默认的文件
        if os.path.exists(updated_file1_path):
            file1_path = updated_file1_path
        else:
            file1_path = default_file1_path

        # 加载文件2: Cells_info.xlsx 的路径
        file2_path = os.path.join(config.cell_classification_output_path, 'Cells_info.xlsx')

        # 检查 Cells_info.xlsx 文件是否存在
        if not os.path.exists(file2_path):
            raise FileNotFoundError(f"Cells_info.xlsx does not exist in the specified path: {file2_path}.")

        # 读取 Excel 文件，读取所有工作表
        file1_data = pd.read_excel(file1_path, sheet_name=None)  # 读取所有工作表
        file2_data = pd.read_excel(file2_path, sheet_name=None)  # 读取所有工作表

        # 打印 file2_data 中的工作表名
        #print(f"Cells_info.xlsx 中的工作表名：{file2_data.keys()}")

        # 获取工作表
        file1_sheet1 = file1_data['Sheet1']  # 读取 file1 中的 Sheet1
        file2_sheet1 = file2_data['Cell']  # 读取 file2 中的 Sheet1

        # 打印 file2_sheet1 中的所有列名
        #print("file2_sheet1 的列名：", file2_sheet1.columns)

        # 检查列名是否包含 '细胞编号'
        if 'Cell Index' not in file2_sheet1.columns:
            raise KeyError(f"'The column 'Cell Index' is missing in 'Cells_info.xlsx'. Available columns are:{file2_sheet1.columns}")

        # 创建输出目录（如果不存在），并清空该目录
        output_dir = os.path.join(config.quantitative_analysis_output_path, output_folder_name)

        # 如果目录存在，删除该目录下的所有文件和子文件夹
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 删除文件或符号链接
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 删除子文件夹
                except Exception as e:
                    print(f"Error occurred while deleting file '{file_path}': {e}")

        # 重新创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 获取总轨迹数
        total_tracks = len(file1_sheet1)

        # 遍历跟踪数据中的每一行
        for index, row in file1_sheet1.iterrows():
            # 提取当前行中的细胞 ID
            cell_ids = row[1:].values  # 提取从第二列开始的所有细胞 ID

            # 获取对应的序号
            sequence_number = row['Index']

            # 创建一个 DataFrame 来存储当前轨迹的匹配行
            track_df = pd.DataFrame()

            # 在细胞信息数据中查找对应的行
            for i, cell_id in enumerate(cell_ids):
                matched_row = file2_sheet1[file2_sheet1['Cell Index'] == cell_id]
                if not matched_row.empty:
                    matched_row = matched_row.copy()
                    # 添加帧编号（1, 2, 3, ...）
                    matched_row.insert(0, 'Index', i + 1)  # 在第一列插入'序号'
                    # 将匹配的行追加到轨迹 DataFrame
                    track_df = pd.concat([track_df, matched_row], ignore_index=True)

            # 将结果保存到新的 Excel 文件
            output_filename = f'track_{sequence_number}_quantitative_analysis.xlsx'
            if not hasattr(self, 'cells_info') or self.cells_info.empty:
                QMessageBox.warning(self, "Notice", "Please extract cell features before performing this operation.")
                return
            output_path = os.path.join(output_dir, output_filename)
            track_df.to_excel(output_path, index=False)

            # 创建并显示进度条文字，更新进度条显示每个细胞的序号
            self.show_progress_bar(f"Quantitative analysis data for cell {sequence_number} has been generated...")

            # 更新进度条
            progress_value = int((index + 1) / total_tracks * 100)
            self.progress_dialog.setValue(progress_value)

        # 返回保存文件的目录
        return output_dir


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
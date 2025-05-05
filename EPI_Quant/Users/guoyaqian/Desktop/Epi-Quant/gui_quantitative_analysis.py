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
from datetime import datetime  # 导入时间模块









# 自定义 MplCanvas 类，加入鼠标事件支持
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
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


class QuantitativeAnalysisGUI(QMainWindow):
    def __init__(self, npy_file_count):
        super().__init__()
        self.setWindowTitle("Quantitative Analysis")
        self.setGeometry(100, 100, 1300, 600)  # 设置窗口大小为1300宽，600高

        self.npy_file_count = npy_file_count

        # 创建中央控件和布局
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建左右布局
        self.bottom_layout = QHBoxLayout()

        # 左侧部分：显示输入框、特征选择、绘制按钮 + 最小编号的图片 + 说明文字
        self.left_layout = QVBoxLayout()

        # 输入布局部分
        input_layout = QHBoxLayout()
        self.label_track = QLabel("Enter Cell ID (comma-separated):")
        input_layout.addWidget(self.label_track)
        self.track_input = QLineEdit(self)
        input_layout.addWidget(self.track_input)

        # 选择特征的下拉框
        self.feature_label = QLabel("Select Features:")
        input_layout.addWidget(self.feature_label)
        self.combo_box = QComboBox(self)
        self.combo_box.addItems([
            'Cell Area', 'Maximum Horizontal Length', 'Maximum Vertical Length', 'Horizontal/Vertical Length Ratio', 'Cell Perimeter', 'Approximate Polygon Vertex Count',
            'Fitted Ellipse Minor Axis', 'Fitted Ellipse Major Axis', 'Ellipse Major/Minor Axis Ratio', 'Fitted Ellipse Angle', 'Circumscribed Circle Radius',
            'Inscribed Circle Radius', 'Center X Coordinate', 'Center Y Coordinate', 'Cell Left Boundary', 'Cell Right Boundary', 'Circularity', 'P-Value', 'leading_edge','MSD'
        ])
        self.combo_box.setFixedWidth(100)  # 设置固定宽度
        input_layout.addWidget(self.combo_box)

        # 将输入框和选择特征部分作为一行添加到左侧布局
        self.left_layout.addLayout(input_layout)

        # 创建绘制按钮
        self.button = QPushButton("Plot", self)
        self.button.setFixedWidth(500)
        self.button.clicked.connect(self.plot_tracks)

        # 将绘制按钮单独放在一行
        self.left_layout.addWidget(self.button)

        # 显示最小编号的图片
        self.min_image_canvas = MplCanvas(self, width=4, height=4, dpi=100)

        # 设置画布的尺寸策略，让它尽可能填充可用空间
        self.min_image_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 将图片按中心对齐，并确保其占据更多空间
        self.left_layout.addWidget(self.min_image_canvas)

        # 创建注释标签
        self.note_label = QLabel("Note: Cell IDs in the yellow box indicate the cell number.")
        self.note_label.setAlignment(Qt.AlignCenter)  # 设置文字居中

        # 将注释添加到左侧布局
        self.left_layout.addWidget(self.note_label)

        # 将左侧布局添加到整体布局中
        self.bottom_layout.addLayout(self.left_layout, 5)  # 左边占5份（对应500像素）

        # 右侧部分：显示图表
        self.right_widget = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_widget)
        self.canvas = FigureCanvas(plt.Figure())
        self.right_layout.addWidget(self.canvas)
        self.bottom_layout.addWidget(self.right_widget, 8)

        # 将左右布局添加到主布局
        main_layout.addLayout(self.bottom_layout)

        self.display_min_image()  # 显示编号最小的图片

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

    def plot_msd_log_log(self):
        """绘制多轨迹的 MSD 对数-对数图及线性拟合，并保存 γ 值到 Excel 文件"""
        if not hasattr(self, 'all_values_dict') or not self.all_values_dict:
            QMessageBox.warning(self, "Invalid Operation",
                                "Please select cells and plot the data before performing MSD calculation.")
            return

        # 弹出输入框，获取时间间隔
        time_interval, ok = QInputDialog.getInt(self, "Enter Time Interval",
                                                "Please enter the time interval (in seconds):", 60, 1, 10000, 1)
        if not ok:
            return  # 用户取消输入时直接返回

        # 创建画布和轴
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        # 使用 tab20 色板（20 种颜色），和足够多的标记样式
        colors = plt.cm.tab20.colors
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+', 'x', 'p', 'H', '8', '<', '>', '|', '_', '.', ',']
        epsilon = 1e-10  # 极小常数，用于避免 log(0) 计算问题

        gamma_values = []  # 用于存储每条轨迹的 γ 值

        # 遍历所有轨迹并绘图
        for idx, (track_number, positions) in enumerate(self.all_values_dict.items()):
            # 提取 x 和 y 坐标
            x_coords = positions['Center X Coordinate'].values
            y_coords = positions['Center Y Coordinate'].values
            msd_values = []
            total_frames = len(x_coords)

            # 遍历不同的时间滞后 (lag) 计算 MSD
            for lag in range(1, total_frames):
                displacements = (x_coords[lag:] - x_coords[:-lag]) ** 2 + (y_coords[lag:] - y_coords[:-lag]) ** 2
                msd = displacements.mean()
                msd_values.append(msd if msd > 0 else epsilon)

            # 对数变换 MSD 和时间间隔
            log_time_lags = np.log(np.array(range(1, total_frames)) * time_interval)
            log_msd = np.log(msd_values)

            # 检查数据是否足够进行拟合
            if len(log_time_lags) <= 1 or len(log_msd) <= 1:
                QMessageBox.warning(self, "Invalid Data",
                                    f"Track {track_number} has insufficient data for MSD calculation.")
                continue

            # 线性拟合
            slope, intercept, r_value, _, _ = linregress(log_time_lags, log_msd)
            gamma_values.append((track_number, slope))

            # 确定颜色和标记，避免重复
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            # 绘制数据点并设置图例标签
            ax.plot(log_time_lags, log_msd, marker=marker, linestyle='None', color=color, markersize=3,
                    label=f'Cell{track_number}, γ = {slope:.2f}')

            # 绘制拟合线，但不添加到图例中
            ax.plot(log_time_lags, intercept + slope * log_time_lags, linestyle='--', color=color, linewidth=1.5)


        # 设置标题和轴标签，统一字体
        ax.set_title(f"Ln(MSD)", fontsize=16, pad=20, color='#2E2E2E')
        ax.set_xlabel("Ln(t)", fontsize=14, labelpad=12, color='#4D4D4D')

        # 设置图例，使用多列布局来节省空间
        ax.legend(loc='upper left', fontsize=9, frameon=True, edgecolor='black', ncol=2, framealpha=0.9)
        ax.grid(True, linestyle=':', color='grey', alpha=0.7)

        self.canvas.draw()

        # 检查 gamma_values 是否为空
        if not gamma_values:
            QMessageBox.warning(self, "No Data", "No gamma values were calculated. Please check your data.")
            return

        # 根据时间戳生成唯一文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化当前时间
        output_dir = config.quantitative_analysis_output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # 创建输出目录

        output_file_name = f'quantitative_analysis_gamma_values_{timestamp}.xlsx'
        output_path = os.path.join(output_dir, output_file_name)

        # 将 gamma_values 转换为 DataFrame
        gamma_df = pd.DataFrame(gamma_values, columns=['Track Number', 'Gamma Value'])
        print(gamma_df)  # 检查生成的 DataFrame

        # 保存 Excel 文件
        try:
            gamma_df.to_excel(output_path, index=False)
            QMessageBox.information(self, "Save Successful", f"Gamma values have been saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"An error occurred while saving gamma values:\n{str(e)}")
            print(f"Error details: {e}")

    def plot_tracks(self):
        # 获取用户输入的轨迹编号
        track_numbers = self.track_input.text().strip().split(',')
        track_numbers = [num.strip() for num in track_numbers if num.strip().isdigit()]

        # 遍历轨迹编号，提取中心点坐标数据
        folders_to_check = [
            'all_cell_quantitative_analysis_output'
        ]

        if not track_numbers:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid cell numbers.")
            return

        selected_column = self.combo_box.currentText()

        # 如果用户选择的是“MSD”，则计算并绘制 MSD 图
        if selected_column == 'MSD':
            # 用于存储轨迹的中心点坐标
            self.all_values_dict = {}

            for track_number in track_numbers:
                track_number = int(track_number)
                file_found = False

                for folder in folders_to_check:
                    file_name = f"track_{track_number}_quantitative_analysis.xlsx"
                    file_path = os.path.join(config.quantitative_analysis_output_path, folder, file_name)

                    if os.path.exists(file_path):
                        # 读取 Excel 文件
                        df = pd.read_excel(file_path)

                        # 检查中心点坐标列是否存在
                        if 'Center X Coordinate' in df.columns and 'Center Y Coordinate' in df.columns:
                            df = df[['Index', 'Center X Coordinate', 'Center Y Coordinate']].dropna()

                            # 使用 x 和 y 坐标进行 MSD 计算
                            # 将每个轨迹的 (x, y) 坐标对存储到字典中
                            self.all_values_dict[track_number] = df.set_index('Index')[['Center X Coordinate', 'Center Y Coordinate']]
                            file_found = True
                            break

                if not file_found:
                    QMessageBox.warning(self, "File Not Found",
                                        f"File track_{track_number}_quantitative_analysis.xlsx does not exist.")
                    return

            # 调用 MSD 绘图函数
            self.plot_msd_log_log()
            # 根据时间戳生成唯一文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化当前时间
            output_file_name = f'Quant_MSD_{timestamp}.pdf'  # 文件名仅包含时间戳
            output_path = os.path.join(config.quantitative_analysis_output_path, output_file_name)

            try:
                self.canvas.figure.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存高分辨率图片
                QMessageBox.information(self, "Save Successful", f"The line chart has been saved to:\n{output_path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", f"An error occurred while saving the line chart:\n{e}")


        else:
            # 设置不同轨迹的颜色
            colors = itertools.cycle(plt.cm.tab10.colors)
            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)

            # 用于存储所有轨迹数据的字典
            self.all_values_dict = {}

            # 设置横坐标的最大值为 npy 文件的数量
            max_frame = self.npy_file_count

            # 依次在文件夹中查找文件
            for track_number in track_numbers:
                track_number = int(track_number)
                file_found = False

                for folder in folders_to_check:
                    file_name = f"track_{track_number}_quantitative_analysis.xlsx"
                    file_path = os.path.join(config.quantitative_analysis_output_path, folder, file_name)

                    if os.path.exists(file_path):
                        # 读取 Excel 文件
                        df = pd.read_excel(file_path)

                        # 检查所选列是否存在
                        if selected_column not in df.columns:
                            QMessageBox.warning(self, "Invalid Data",
                                                f"The selected column {selected_column}  does not exist in the file  {file_name} .")
                            return

                        # 使用 dropna() 忽略空值，并将轨迹数据存储到字典中
                        df = df[['Index', selected_column]].dropna()
                        self.all_values_dict[track_number] = df.set_index('Index')[selected_column]

                        # 绘制轨迹，跳过空值的点
                        color = next(colors)
                        ax.plot(df['Index'], df[selected_column], marker=None, linestyle='-', color=color, linewidth=1,
                                label=f'Cell{track_number}')

                        file_found = True
                        break


                if not file_found:
                    QMessageBox.warning(self, "File Not Found",
                                        f"File track_{track_number}_quantitative_analysis.xlsx does not exist.")
                    return

            # 创建一个 DataFrame 来对齐不同长度的数据
            if self.all_values_dict:
                aligned_df = pd.DataFrame(self.all_values_dict).reindex(range(1, max_frame + 1))  # 重新索引到最大帧数

                # 去掉任何含有 NaN 的行（即帧）
                #aligned_df.dropna(axis=0, inplace=True)

                # 计算剩余帧的平均值
                average_values = aligned_df.mean(axis=1)

                # 绘制平均值曲线
                ax.plot(average_values.index, average_values.values, marker=None, linestyle='-', color='red',
                        linewidth=2,
                        label='Mean')

            # 设置横坐标范围，统一到最大帧数（即 npy 文件的数量）
            ax.set_xlim(0, max_frame)

            # 确保横坐标显示整数刻度
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))



            # 设置标题和轴标签，统一字体
            ax.set_title(f"{selected_column}", fontsize=16, pad=20, color='#2E2E2E')
            ax.set_xlabel("Frame", fontsize=14, labelpad=12, color='#4D4D4D')

            # 添加y轴虚线
            ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
            ax.set_facecolor('white')

            # 设置粗体的坐标轴线条
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=10)

            # 添加图例以区分不同的轨迹和平均值
            ax.legend(loc='best', fontsize=10)

            # 根据时间戳生成唯一文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化当前时间
            output_file_name = f'Quant_{selected_column}_{timestamp}.pdf'  # 文件名包含时间戳和所选列
            output_path = os.path.join(config.quantitative_analysis_output_path, output_file_name)

            try:
                self.canvas.figure.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存高分辨率图片
                QMessageBox.information(self, "Save Successful", f"The line chart has been saved to:\n{output_path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", f"An error occurred while saving the line chart:\n{e}")

            # 使用紧凑布局避免重叠
            self.canvas.draw()



if __name__ == '__main__':
    # 假设 npy_file_count 的值为100，根据实际数据情况修改
    npy_file_count = 100

    app = QApplication(sys.argv)
    main_window = QuantitativeAnalysisGUI(npy_file_count)
    main_window.show()
    sys.exit(app.exec_())

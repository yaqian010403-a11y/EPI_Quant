import os

# 初始化所有路径变量为 None
npy_folder_path = None
Img_path = None
cell_classification_output_path = None
cell_track_output_path = None
quantitative_analysis_output_path = None

def set_paths(npy_path, img_path, path):
    global npy_folder_path, Img_path, cell_classification_output_path, cell_track_output_path, quantitative_analysis_output_path

    # 设置基本路径
    npy_folder_path = npy_path
    Img_path = img_path

    # 根据输出文件路径创建或设置其他路径
    cell_classification_output_path = os.path.join(path, 'cell_classification_output')
    cell_track_output_path = os.path.join(path, 'cell_track_output')
    quantitative_analysis_output_path = os.path.join(path, 'quantitative_analysis_output')

    # 确保这些目录存在，如果不存在则创建
    os.makedirs(cell_classification_output_path, exist_ok=True)
    os.makedirs(cell_track_output_path, exist_ok=True)
    os.makedirs(quantitative_analysis_output_path, exist_ok=True)

    # 可以在此处添加更多路径设置


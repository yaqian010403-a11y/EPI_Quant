import os

# Initialize all path variables to None
npy_folder_path = None
Img_path = None
cell_classification_output_path = None
cell_track_output_path = None
quantitative_analysis_output_path = None

def set_paths(npy_path, img_path, path):
    global npy_folder_path, Img_path, cell_classification_output_path, cell_track_output_path, quantitative_analysis_output_path

    # Set basic paths
    npy_folder_path = npy_path
    Img_path = img_path

    # Create or set other paths based on the output file path
    cell_classification_output_path = os.path.join(path, 'cell_classification_output')
    cell_track_output_path = os.path.join(path, 'cell_track_output')
    quantitative_analysis_output_path = os.path.join(path, 'quantitative_analysis_output')

    # Ensure these directories exist, create them if they do not
    os.makedirs(cell_classification_output_path, exist_ok=True)
    os.makedirs(cell_track_output_path, exist_ok=True)
    os.makedirs(quantitative_analysis_output_path, exist_ok=True)

    # Additional path settings can be added here


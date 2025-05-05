import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import utils
import re
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

import config

def load_and_process_data(file_path):
    # 加载Excel文件
    data = pd.read_excel(file_path)
    # 删除不需要的“差异度”列
    data.drop(columns=['差异度'], inplace=True)
    return data

def main():
    xlsx_file = os.path.join(config.output_path, 'Cells_info.xlsx')
    path = os.path.join(config.path, 'cell_track_output/')
    base_path = os.path.join(config.path, 'cell_track_output/Cell_Matching_Results_new_')
    cells_info = pd.read_excel(xlsx_file, index_col=[0])

    pd.set_option('display.max_row', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 180)

    fig_id_all = [i for i in range(1, 20)]
    for fig_id in fig_id_all:
        cells_info_fig1 = cells_info[cells_info.index.str.contains(f'细胞{fig_id}_')]
        cells_info_fig2 = cells_info[cells_info.index.str.contains(f'细胞{fig_id + 1}_')]
        cells_info_fig1_index = list(cells_info_fig1.index)
        cells_info_fig2_index = list(cells_info_fig2.index)

        cells_info_fig1_positive = cells_info_fig1[cells_info_fig1['leading_edge'] >= 0]
        mean_AP_fig1 = cells_info_fig1_positive['AP'].mean()

        print(f"Frame {fig_id}: Average AP (excluding negative leading_edge) = {mean_AP_fig1}")

        cells_relation = np.zeros((len(cells_info_fig1_index), len(cells_info_fig2_index)))
        cells_nature = ['面积', 'AP/DV', '周长', '近似多边形顶点数',
                        '拟合椭圆短轴', '拟合椭圆长轴', '拟合椭圆角度',
                        '外接圆半径', 'AP', 'DV', '中心点坐标1', '中心点坐标2', 'leading_edge']
        cells_weight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
        cell1_cell2_distance = 0
        for cell1_id in range(len(cells_info_fig1_index)):
            cell1_id_index = cells_info_fig1_index[cell1_id]
            cell1_info = list(cells_info_fig1.loc[cell1_id_index, cells_nature])
            for cell2_id in range(len(cells_info_fig2_index)):
                cell2_id_index = cells_info_fig2_index[cell2_id]
                cell2_info = list(cells_info_fig2.loc[cell2_id_index, cells_nature])
                max_distance = max(cell1_info[8], cell1_info[9])

                if abs(cell2_info[10] - cell1_info[10]) > max_distance or abs(
                        cell2_info[11] - cell1_info[11]) > max_distance or cell1_info[12] * cell2_info[12] < 0:
                    cells_relation[cell1_id, cell2_id] = np.nan
                    continue
                for cell_nature_id in range(len(cells_nature)):
                    cell1_cell2_distance += cells_weight[cell_nature_id] * (
                            cell2_info[cell_nature_id] - cell1_info[cell_nature_id]) ** 2
                cell1_cell2_distance = np.sqrt(cell1_cell2_distance)
                cells_relation[cell1_id, cell2_id] = cell1_cell2_distance
                cell1_cell2_distance = 0

        df = pd.DataFrame(cells_relation, index=cells_info_fig1_index, columns=cells_info_fig2_index)
        df.to_excel(os.path.join(path, "cells_relation.xlsx"))

        cells_track_fig = pd.DataFrame(
            columns=[f'图像{fig_id}', f'图像{fig_id + 1}', '相似度'])

        file_path = os.path.join(path, "cells_relation.xlsx")
        df1 = pd.read_excel(file_path, header=0, index_col=0)

        max_value = 23
        df_filled = df1.fillna(max_value)

        rows, cols = df_filled.shape
        if rows != cols:
            size = max(rows, cols)
            new_df = pd.DataFrame(max_value, index=pd.Index(range(size), name=df.index.name),
                                  columns=pd.Index(range(size), name=df.columns.name))
            new_df.iloc[:rows, :cols] = df_filled
            df_filled = new_df

        row_ind, col_ind = linear_sum_assignment(df_filled.values)

        matching_result = pd.DataFrame({
            f'图像{fig_id}': [df.index[r] if r < len(df.index) else f"细胞{fig_id}_" for r in row_ind],
            f'图像{fig_id + 1}': [df.columns[c] if c < len(df.columns) else f"细胞{fig_id + 1}_" for c in col_ind],
            '差异度': [df_filled.iat[r, c] for r, c in zip(row_ind, col_ind)]
        })

        matching_result.insert(0, '序号', range(len(matching_result)))

        filtered_result = matching_result.dropna(subset=['差异度'])
        filtered_result = filtered_result[filtered_result['差异度'] < max_value]

        filtered_output_file_path = f'{base_path}{fig_id}_{fig_id + 1}.xlsx'
        filtered_result.to_excel(filtered_output_file_path, index=False)

    # 自动确定文件数量
    files = [f for f in os.listdir(path) if re.match(r'Cell_Matching_Results_new_\d+_\d+\.xlsx', f)]
    files_count = len(files) + 1  # 文件数等于最大编号+1

    # 初始化，加载第一个文件
    data_final = load_and_process_data(f'{base_path}1_2.xlsx')
    data_final.columns = ['序号', '图像1', '图像2']

    # 循环处理剩下的文件
    for i in range(2, files_count - 1):
        next_file_path = f'{base_path}{i}_{i + 1}.xlsx'
        data_next = load_and_process_data(next_file_path)
        data_next.columns = ['序号', f'图像{i}', f'图像{i + 1}']

        data_final[f'图像{i + 1}'] = None

        for index, row in data_next.iterrows():
            match_index = data_final[data_final[f'图像{i}'] == row[f'图像{i}']].index
            if not match_index.empty:
                data_final.at[match_index[0], f'图像{i + 1}'] = row[f'图像{i + 1}']

    print(data_final)

    output_path = os.path.join(config.path, 'cell_track_output/', 'Merged_Cell_Tracking_Results.xlsx')
    data_final.to_excel(output_path, index=False)

    print(f"合并后的数据已保存到 {output_path}")

    data_final_cleaned = data_final.dropna()

    clean_output_path = os.path.join(config.path, 'cell_track_output/', 'Cleaned_Merged_Cell_Tracking_Results.xlsx')
    data_final_cleaned.to_excel(clean_output_path, index=False)

    print(f"清理后的数据已保存到 {clean_output_path}")

    cells_track = pd.read_excel(clean_output_path, index_col=[0])
    regex = re.compile(r'\d+')

    fig_id_all = list(cells_track.columns)

    for (root, dirs, files) in os.walk(path):
        for filename in files:
            basename, ext = os.path.splitext(filename)
            if ext != '.npy':
                continue

            dat = np.load(os.path.join(path, filename), allow_pickle=True).item()
            cell_contours = utils.outlines_list(dat['masks'])

            fig_id = int(max(regex.findall(filename)))

            if f'图像{fig_id}' not in fig_id_all:
                continue
            cell_id = 0
            for cell_contour in cell_contours:
                cell_id += 1

                if f'细胞{fig_id}_{cell_id}' in list(cells_track.loc[:, f'图像{fig_id}']):
                    cell_id_index = cells_track[
                        cells_track[f'图像{fig_id}'] == f'细胞{fig_id}_{cell_id}'].index[0]

                    cx = int(0.5 * np.max(cell_contour[:, 0]) + 0.5 * np.min(cell_contour[:, 0]))
                    cy = int(0.5 * np.max(cell_contour[:, 1]) + 0.5 * np.min(cell_contour[:, 1]))
                    plt.text(cx, cy, str(cell_id_index), fontsize=7)
                    plt.plot(cell_contour[:, 0], cell_contour[:, 1], 'r')

            plt.imshow(dat['img'])
            plt.colorbar()
            plt.title(filename)

            normalization_folder = 'show_cells_track'
            normalization_path = os.path.join(path, normalization_folder)

            if not os.path.exists(normalization_path):
                os.makedirs(normalization_path)

            file_name = f'{fig_id}_leading_edge.png'
            file_path = os.path.join(normalization_path, file_name)

            plt.savefig(file_path)
            plt.close()

if __name__ == '__main__':
    main()

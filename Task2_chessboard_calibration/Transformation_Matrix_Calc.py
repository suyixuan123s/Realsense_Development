"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: Transformation_Matrix_Calc.py
Description:

"""

import numpy as np


def get_transformation_matrix():
    # 旋转矩阵 R
    R = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, -1]])

    # 平移向量 t
    t = np.array([[0.41],
                  [0.050],
                  [0.011]])

    # 构造齐次变换矩阵 T
    T = np.eye(4)  # 创建一个 4x4 的单位矩阵
    T[:3, :3] = R  # 设置旋转部分
    T[:3, 3] = t.flatten()  # 设置平移部分

    return T


def get_camera_to_table_transformation(camera_to_board, board_to_table):
    # 从相机坐标系到桌面坐标系的转换关系
    camera_to_table = np.dot(board_to_table, camera_to_board)
    return camera_to_table


if __name__ == "__main__":
    # 相机坐标系到棋盘格坐标系的变换矩阵
    camera_to_board = np.array([[-0.0193798, -0.99946704, 0.0262689, 0.53903028],
                                [0.84030241, -0.00204524, 0.54211409, 0.15331288],
                                [-0.54177144, 0.03257988, 0.8398942, 1.44357683],
                                [0., 0., 0., 1.]])

    # 棋盘格坐标系到桌面坐标系的变换矩阵
    board_to_table = get_transformation_matrix()

    # 计算相机坐标系到桌面坐标系的变换矩阵
    camera_to_table = get_camera_to_table_transformation(camera_to_board, board_to_table)

    # 打印结果
    print("Transformation Matrix from Camera Coordinate System to Table Coordinate System:")
    print(camera_to_table)

    # 保存变换矩阵到文件
    file_name = "camera_to_table_transformation_matrix.txt"
    with open(file_name, "w") as f:
        f.write("Transformation Matrix from Camera Coordinate System to Table Coordinate System:\n")
        f.write(str(camera_to_table))
    print(f"Transformation matrix saved to {file_name}")

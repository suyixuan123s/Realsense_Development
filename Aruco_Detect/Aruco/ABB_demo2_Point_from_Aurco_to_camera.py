"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import numpy as np

# 假设您已经得到了旋转矩阵和平移向量
rotation_matrix = np.array([[0.93101263, 0.36363439, 0.03139285],
                            [0.36363439, -0.91672725, -0.16547261],
                            [-0.03139285, 0.16547261, -0.98571462]])

translation_vector = np.array([0.08560594, -0.02608648, 0.39376504])

# 定义平面上的任意一点
point_on_plane = np.array([0, 0, 0])  # 替换为实际的平面坐标

# 将平面上的点转换到相机坐标系
point_in_camera = rotation_matrix @ point_on_plane + translation_vector

print("Point in Camera Coordinate System:", point_in_camera)




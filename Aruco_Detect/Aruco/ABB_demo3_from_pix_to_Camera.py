"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import cv2
import numpy as np

rotation_matrix = np.array([[0.93101263, 0.36363439, 0.03139285],
                            [0.36363439, -0.91672725, -0.16547261],
                            [-0.03139285, 0.16547261, -0.98571462]])
translation_vector = np.array([0.08560594, -0.02608648, 0.39376504])

# 设置相机内参和畸变系数
fx = 434.43981934
fy = 433.24884033
cx = 322.23464966
cy = 236.84153748

camera_matrix = np.array([[434.43981934, 0, 322.23464966],
                          [0, 433.24884033, 236.84153748],
                          [0, 0, 1]])
k1 = -0.05277087
k2 = 0.06000207
p1 = 0.00087849
p2 = 0.00136543
k3 = -0.01997724
depth_scale = 9.999999747378752e-05

dist_coeffs = np.array([k1, k2, p1, p2, k3])

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# 物体在图像中的像素坐标
u, v = 338, 241

# 将像素坐标转换为归一化坐标
x_norm = (u - cx) / fx
y_norm = (v - cy) / fy


# ------------- 深度值的计算 -----------
depth_map = cv2.imread('../../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/depth_image_20250409-165409.png',
                       cv2.IMREAD_UNCHANGED)  # 读取深度图
depth_value = depth_map[v, u] * depth_scale  # 获取该像素的深度值
print(f"Depth value at pixel (u, v)({u}, {v}) = {depth_value}")
# 计算物体在平面上的三维坐标
P_object_plane = np.array([x_norm * depth_value, y_norm * depth_value, depth_value])
# 将物体的位置转换到相机坐标系
P_object_camera1 = rotation_matrix @ P_object_plane + translation_vector
print("Object Position in Camera Coordinate System:", P_object_camera1)
Average_Transformation_Matrix = np.array([[-0.49262632, -0.50407491, -0.0583373, 0.00643042],
                                          [-0.50235654, 0.49289276, -0.09278652, 0.00183109],
                                          [0.05650777, -0.00889072, -0.99237892, 0.16757868],
                                          [0., 0., 0., 1.]])

P_object_plane_homogeneous = np.append(P_object_plane, 1)
P_object_camera2 = Average_Transformation_Matrix @ P_object_plane_homogeneous
# 获取前三个数（即3D坐标）
P_object_camera2_3D = P_object_camera2[:3]
# 设置NumPy打印选项以禁用科学计数法
np.set_printoptions(suppress=True)
print("平均:", P_object_camera2_3D)

"""
Author: Yixuan Su
Date: 2024/11/20 10:36
File: Analyze_segmented_object_point_cloud.py
Description:
"""

import open3d as o3d
import numpy as np

# 点云文件路径
point_cloud_path = r'E:\ABB\segment-anything\realsense_yolov9_simulation_detection\SAM277\centrifuge_close\centrifuge_close_0\point_cloud.ply'

# 加载点云文件（保留颜色信息）
point_cloud = o3d.io.read_point_cloud(point_cloud_path)

# 检查点云是否包含颜色信息
if point_cloud.has_colors():
    print("点云包含颜色信息，将显示带颜色的点云。")
else:
    print("点云不包含颜色信息，将显示为无色点云。")

# 获取点云的所有点坐标和颜色
points = np.asarray(point_cloud.points)
colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None

# 计算点的数量
num_points = points.shape[0]

# 计算中心点（所有点的平均坐标）
center_point = points.mean(axis=0)

# 定义从仿真坐标系到相机坐标系的外参
alpha, beta, gamma = 148.0, 0.4, 178.0  # 旋转角度
tx, ty, tz = -0.525, -0.76, -1.25  # 平移


def get_transformation_matrix(alpha, beta, gamma, tx, ty, tz):
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]
    return transformation_matrix


# 获取转换矩阵
transformation_matrix_sim_to_cam = get_transformation_matrix(alpha, beta, gamma, tx, ty, tz)

# 将中心点转换到齐次坐标系（添加1作为第四维）
center_point_homogeneous = np.append(center_point, 1)

# 使用变换矩阵将中心点转换到相机坐标系
camera_coords_center = transformation_matrix_sim_to_cam @ center_point_homogeneous

# 提取转换后的3D坐标
camera_coords_center = camera_coords_center[:3]  # 提取前三维

# 创建一个球体作为中心点高亮显示
center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 设置球体的半径
center_sphere.translate(center_point)  # 将球体移动到中心点位置
center_sphere.paint_uniform_color([1, 0, 0])  # 将球体颜色设为红色

# 将原始点和球体一起显示
o3d.visualization.draw_geometries([point_cloud, center_sphere], window_name="Point Cloud with Center Highlight",
                                  width=800, height=600)

# 输出结果
print(f"点的数量: {num_points}")
print(f"中心点在仿真坐标系下的位置: {center_point}")
print(f"中心点在相机坐标系下的位置: {camera_coords_center}")

# 返回点的数量、中心点（仿真坐标系）、中心点（相机坐标系）和颜色信息（如果存在）
num_points, center_point, camera_coords_center, colors

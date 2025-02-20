"""
Author: Yixuan Su
Date: 2024/11/20 10:36
File: Point_cloud_visualazation_size.py
Description:
"""

import open3d as o3d
import numpy as np

# 加载点云文件
pcd1 = o3d.io.read_point_cloud("E:\ABB\AI\Depth-Anything-V2\Point_cloud_files\demo24\colored_point_cloud.ply")
pcd2 = o3d.io.read_point_cloud("E:\ABB\AI\Depth-Anything-V2\metric_depth\output1\color_image_20241024-191620.ply")

# 获取两个点云的包围盒，并获取其尺寸
bbox1 = pcd1.get_axis_aligned_bounding_box()
bbox2 = pcd2.get_axis_aligned_bounding_box()

# 获取两个点云的包围盒尺寸
size1 = np.array(bbox1.get_extent())  # pcd1 的尺寸
size2 = np.array(bbox2.get_extent())  # pcd2 的尺寸

# 使用第二个点云的最大尺寸来统一缩放点云1
max_size_pcd2 = max(size2)
max_size_pcd1 = max(size1)

# 计算缩放比例，使第一个点云与第二个点云的最大尺寸一致
scale_factor = max_size_pcd2 / max_size_pcd1
pcd1.scale(scale_factor, center=pcd1.get_center())  # 按照点云2的最大尺寸缩放点云1

# 缩放之后重新计算两个点云的中心点
center1 = pcd1.get_center()
center2 = pcd2.get_center()

# 计算平移量，将第一个点云的中心平移到与第二个点云的中心对齐
translation = center2 - center1
pcd1.translate(translation)

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Clouds Aligned to Size of Point Cloud 2")

# 将两个点云添加到同一个窗口
vis.add_geometry(pcd1)
vis.add_geometry(pcd2)

# 运行可视化
vis.run()
vis.destroy_window()

# '''
# 两点云文件上下显示
# '''
#
# import open3d as o3d
# import numpy as np
#
# # 加载点云文件
# pcd1 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\Point_cloud_Datasets\\demo24\\colored_point_cloud.ply")
# pcd2 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\metric_depth\\output1\\color_image_20241024-191620.ply")
#
# # 获取两个点云的包围盒，并获取其尺寸
# bbox1 = pcd1.get_axis_aligned_bounding_box()
#
# print(bbox1)
#
# bbox2 = pcd2.get_axis_aligned_bounding_box()
#
# print(bbox2)
# # 获取两个点云的包围盒尺寸
# size1 = np.array(bbox1.get_extent())  # pcd1 的尺寸
# size2 = np.array(bbox2.get_extent())  # pcd2 的尺寸
#
# # 使用第一个点云的最大尺寸来统一缩放点云二
# max_size_pcd1 = max(size1)
# max_size_pcd2 = max(size2)
#
# # 计算缩放比例，使第二个点云与第一个点云的最大尺寸一致
# scale_factor = max_size_pcd1 / max_size_pcd2
# pcd2.scale(scale_factor, center=pcd2.get_center())  # 按照点云1的最大尺寸缩放点云2
#
# # # 缩放后重新计算 bbox2，因为 pcd2 已经被缩放
# # bbox2_scaled = pcd2.get_axis_aligned_bounding_box()
# #
# # # 获取缩放后的 z 轴高度
# # height_pcd2_scaled = bbox2_scaled.get_extent()[2]
# # print(height_pcd2_scaled)
#
# # 缩放之后重新计算两个点云的中心点
# center1 = pcd1.get_center()
#
# print(center1)
# center2 = pcd2.get_center()
# print(center2)
#
#
#
# # 获取两个点云的包围盒的高度 (z 方向或 y 方向)
# height_pcd1 = bbox1.get_extent()[2]  # 选择z轴高度
# print(f"height_pcd1:{bbox1.get_extent()[2]}")
#
# height_pcd2 = bbox2.get_extent()[2]
#
# print(f"height_pcd2:{bbox2.get_extent()[2]}")
#
# # 将点云1向下平移，使其在点云2的下方显示
# translation = np.array([0, 0, -(height_pcd1 + height_pcd2) / 2])  # 调整z方向使其上下分布
# pcd1.translate(translation)
#
#
# # # 计算平移量，将第一个点云的中心平移到与第二个点云的中心对齐
# # translation = center2 - center1
# # pcd1.translate(translation)
#
# # 创建可视化窗口
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Point Clouds Aligned Vertically")
#
# # 将两个点云添加到同一个窗口
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd2)
#
# # 运行可视化
# vis.run()
# vis.destroy_window()

#
# import open3d as o3d
# import numpy as np
#
# # 加载点云文件
# pcd1 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\Point_cloud_Datasets\\demo24\\colored_point_cloud.ply")
# pcd2 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\metric_depth\\output1\\color_image_20241024-191620.ply")
#
# # 获取两个点云的包围盒，并获取其尺寸
# bbox1 = pcd1.get_axis_aligned_bounding_box()
# bbox2 = pcd2.get_axis_aligned_bounding_box()
#
# # 获取两个点云的包围盒尺寸
# size1 = np.array(bbox1.get_extent())  # pcd1 的尺寸
# size2 = np.array(bbox2.get_extent())  # pcd2 的尺寸
#
# # 分别计算 x、y、z 方向的缩放因子
# scale_factors = size1 / size2
#
# # 对点云2进行非等比例缩放，使其包围盒在所有方向上与点云1相同
# pcd2.scale(scale_factors[0], center=pcd2.get_center())  # x 方向缩放
# pcd2.scale(scale_factors[1], center=pcd2.get_center())  # y 方向缩放
# pcd2.scale(scale_factors[2], center=pcd2.get_center())  # z 方向缩放
#
# # 获取两个点云的包围盒的高度 (z 方向)
# height_pcd1 = bbox1.get_extent()[2]  # 选择z轴高度
# height_pcd2 = bbox2.get_extent()[2]  # 选择z轴高度
#
# # 将点云1向下平移，使其在点云2的下方显示
# translation = np.array([0, 0, -(height_pcd1 + height_pcd2)])  # 调整z方向使其上下分布
# pcd1.translate(translation)
#
# # 创建可视化窗口
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Point Clouds Aligned Vertically")
#
# # 将两个点云添加到同一个窗口
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd2)
#
# # 运行可视化
# vis.run()
# vis.destroy_window()

#


#
# '''
#
# 点云的上下展示，实现对每个方向的非等比例缩放
#
# '''
#
# import open3d as o3d
# import numpy as np
#
# # 加载点云文件
# pcd1 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\Point_cloud_Datasets\\demo24\\colored_point_cloud.ply")
# pcd2 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\metric_depth\\output1\\color_image_20241024-191620.ply")
#
# # 获取两个点云的包围盒，并获取其尺寸
# bbox1 = pcd1.get_axis_aligned_bounding_box()
# bbox2 = pcd2.get_axis_aligned_bounding_box()
#
# # 获取两个点云的包围盒尺寸
# size1 = np.array(bbox1.get_extent())  # pcd1 的尺寸
# size2 = np.array(bbox2.get_extent())  # pcd2 的尺寸
#
# # 分别计算 x、y、z 方向的缩放因子
# scale_factors = size1 / size2
#
# # 对点云2进行非等比例缩放
# points = np.asarray(pcd2.points)
# points[:, 0] *= scale_factors[0]  # x 方向缩放
# points[:, 1] *= scale_factors[1]  # y 方向缩放
# points[:, 2] *= scale_factors[2]  # z 方向缩放
# pcd2.points = o3d.utility.Vector3dVector(points)
#
# # 获取两个点云的包围盒的高度 (z 方向)
# height_pcd1 = bbox1.get_extent()[2]  # 选择z轴高度
# height_pcd2 = bbox2.get_extent()[2]  # 选择z轴高度
#
# # 将点云1向下平移，使其在点云2的下方显示
# translation = np.array([0, 0, -(height_pcd1 + height_pcd2)])  # 调整z方向使其上下分布
#
# pcd1.translate(translation)
#
# # 创建可视化窗口
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Point Clouds Aligned Vertically")
#
# # 将两个点云添加到同一个窗口
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd2)
#
# # 运行可视化
# vis.run()
# vis.destroy_window()


#
# '''
# 点云的上下展示，实现对每个方向的非等比例缩放，并保持质心不动
# '''
#
# import open3d as o3d
# import numpy as np
#
# # 加载点云文件
# pcd1 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\Point_cloud_Datasets\\demo24\\colored_point_cloud.ply")
# pcd2 = o3d.io.read_point_cloud("E:\\ABB\\AI\\Depth-Anything-V2\\metric_depth\\output1\\color_image_20241024-191620.ply")
#
# # 获取两个点云的包围盒，并获取其尺寸
# bbox1 = pcd1.get_axis_aligned_bounding_box()
# bbox2 = pcd2.get_axis_aligned_bounding_box()
#
# # 获取两个点云的包围盒尺寸
# size1 = np.array(bbox1.get_extent())  # pcd1 的尺寸
# size2 = np.array(bbox2.get_extent())  # pcd2 的尺寸
#
# # 分别计算 x、y、z 方向的缩放因子
# scale_factors = size1 / size2
#
# # 对点云2进行非等比例缩放，保持质心不动
# points = np.asarray(pcd2.points)
#
# # 计算质心
# centroid = np.mean(points, axis=0)
#
# # 平移点云2，使质心位于原点
# points -= centroid
#
# # 对点云2进行非等比例缩放
# points[:, 0] *= scale_factors[0]  # x 方向缩放
# points[:, 1] *= scale_factors[1]  # y 方向缩放
# points[:, 2] *= scale_factors[2]  # z 方向缩放
#
# # 将点云2移回原来的质心位置
# points += centroid
#
# # 更新点云
# pcd2.points = o3d.utility.Vector3dVector(points)
#
# # 获取两个点云的包围盒的高度 (z 方向)
# height_pcd1 = bbox1.get_extent()[2]  # 选择z轴高度
# height_pcd2 = bbox2.get_extent()[2]  # 选择z轴高度
#
# # 将点云1向下平移，使其在点云2的下方显示
# translation = np.array([0, 0, -(height_pcd1 + height_pcd2)])  # 调整z方向使其上下分布
# pcd1.translate(translation)
#
# # 创建可视化窗口
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Point Clouds Aligned Vertically")
#
# # 将两个点云添加到同一个窗口
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd2)
#
# # 运行可视化
# vis.run()
# vis.destroy_window()

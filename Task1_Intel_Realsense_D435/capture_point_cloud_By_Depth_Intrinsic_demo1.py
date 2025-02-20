"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: capture_point_cloud_By_Depth_Intrinsic_demo1.py
Description:

"""

import os
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import time

# 定义保存图像的目录
save_directory = r'E:\ABB-Project\ABB_wrs\suyixuan\ABB\depth_anything_v2\Point_cloud_Dataset'

# 确保目录存在
os.makedirs(save_directory, exist_ok=True)


def generate_colored_point_cloud(color_image, depth_image, intrinsic_matrix):
    # 获取图像的尺寸，深度图是二维的 (height: 高度, width: 宽度)
    height, width = depth_image.shape

    # 创建一个空的 Open3D_Basic_Edition 点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 从内参矩阵中提取相机的焦距和光心位置 (fx, fy 是焦距, cx, cy 是光心)
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # 将 BGR 彩色图像转换为 RGB 格式（因为 Open3D_Basic_Edition 需要 RGB 格式）
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 初始化用于存储点的 3D 坐标和颜色的列表
    points = []
    colors = []

    # 遍历每个像素 (v: 行, u: 列)
    for v in range(height):
        for u in range(width):
            # 获取深度值，深度值通常是以毫米为单位，乘以 0.001 转换为米
            depth = depth_image[v, u] * 0.001
            if depth > 0:  # 如果深度值大于 0，则计算 3D 坐标
                # 计算 3D 坐标 (X, Y, Z)
                z = depth  # Z 轴为深度值
                x = (u - cx) * z / fx  # X 轴由像素坐标和深度计算得出
                y = (v - cy) * z / fy  # Y 轴由像素坐标和深度计算得出
                points.append([x, y, z])  # 将计算出的 3D 点加入点列表

                # 获取对应点的颜色值（已经转换为 RGB 格式）
                color = color_image[v, u] / 255.0  # 将颜色值归一化到 [0, 1] 范围
                colors.append(color)  # 将颜色加入颜色列表

    # 将计算出的 3D 点赋值给点云对象
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    # 将计算出的颜色赋值给点云对象
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    return point_cloud  # 返回生成的点云对象


# 初始化 RealSense 管道对象
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色和深度流，640x480 分辨率，30 帧率
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 启用深度流，分辨率640x480，帧率30fps
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 启用彩色流，格式为BGR8，帧率30fps

# 启动 RealSense 管道并开始数据流
pipeline.start(config)

# 设置对齐器，用于将深度图像与彩色图像对齐
align = rs.align(rs.stream.color)

try:
    while True:
        # 获取相机的帧数据（深度图像和彩色图像）1
        frames = pipeline.wait_for_frames()

        # 对齐深度图像到彩色图像
        aligned_frames = align.process(frames)

        # 获取对齐后的彩色帧和深度帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # 检查是否成功获取到彩色帧和深度帧
        if not color_frame or not depth_frame:
            continue  # 如果没有成功获取，则跳过这一帧

        # 将彩色帧转换为 NumPy 数组，用于后续处理和显示
        color_image = np.asanyarray(color_frame.get_data())

        # 在窗口中显示彩色图像
        cv2.imshow('Realsense', color_image)

        # 检查是否按下键盘输入
        key = cv2.waitKey(1)

        # 如果按下了 Enter 键 (键码为 13)，则生成点云
        if key == 13:

            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # 将深度帧转换为 NumPy 数组
            depth_image = np.asanyarray(depth_frame.get_data())

            # 获取相机的内参，用于将像素坐标转换为相机坐标系
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            intrinsic_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                         [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                         [0, 0, 1]])

            # 调用函数生成带有颜色信息的 3D 点云
            point_cloud = generate_colored_point_cloud(color_image, depth_image, intrinsic_matrix)

            point_cloud_path = os.path.join(save_directory, f'color_image_{timestamp}.ply')

            # save_path = r'E:\ABB\AI\Depth-Anything-V2\suyixuan\Realsense_Point_cloud_Datasets\colored_point_cloud1203.ply'
            # save_directory = os.path.dirname(save_path)  # 获取保存路径的目录部分

            # 检查并创建目录
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # 将点云保存为 PLY 文件
            o3d.io.write_point_cloud(point_cloud_path, point_cloud)
            print(f"点云已保存为 {point_cloud_path}")

            break  # 生成点云后退出循环

finally:
    # 结束管道并释放资源
    pipeline.stop()
    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()

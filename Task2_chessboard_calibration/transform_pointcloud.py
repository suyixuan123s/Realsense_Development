"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: transform_pointcloud.py
Description:

"""

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from Transformation_Matrix_Camera_to_Chessboard import calibrate_camera


def transform_point_cloud(pcd, transformation_matrix):
    """
    将点云从相机坐标系转换到桌面坐标系。
    :param pcd: 点云对象 (open3d.geometry.PointCloud)
    :param transformation_matrix: 相机到桌面的齐次变换矩阵 (4x4)
    :return: 转换后的点云对象
    """
    # 获取点云的所有点
    points = np.asarray(pcd.points)
    ones_column = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones_column)).T  # 变为齐次坐标，4xN

    # 应用变换矩阵
    transformed_points_homogeneous = np.dot(transformation_matrix, points_homogeneous)
    transformed_points = transformed_points_homogeneous[:3, :].T  # 转换回 3D 坐标

    # 创建新的点云对象
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    # 保留颜色
    if pcd.has_colors():
        transformed_pcd.colors = pcd.colors

    return transformed_pcd


# 示例使用
if __name__ == '__main__':
    # 调用相机标定函数，获取相机的内参、畸变系数和变换矩阵
    chessboard_size = (11, 8)
    square_size = 0.03  # 每个格子的大小为 0.03 米（30 毫米）
    image_directory = 'E:\\ABB-Project\\cc-wrs\\ABB_Intel_Realsense\\Dataset'
    camera_matrix, dist_coeffs, transformation_matrices = calibrate_camera(chessboard_size, square_size,
                                                                           image_directory)

    # 配置 RealSense 管道以采集点云数据
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    # 获取帧并对齐深度和彩色帧
    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("未能获取深度帧或彩色帧")
        pipeline.stop()
    else:
        # 转换深度帧为点云
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_image),
            o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy,
                                              intrinsics.ppx, intrinsics.ppy)
        )

        # 给点云添加颜色
        pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

        # 应用变换矩阵，将点云从相机坐标系转换到桌面坐标系
        transformation_matrix = transformation_matrices[0]  # 使用第一张图像的变换矩阵作为示例
        transformed_pcd = transform_point_cloud(pcd, transformation_matrix)

        # 可视化转换后的点云
        o3d.visualization.draw_geometries([transformed_pcd])

    pipeline.stop()

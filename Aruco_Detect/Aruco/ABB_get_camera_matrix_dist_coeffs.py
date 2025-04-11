"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import pyrealsense2 as rs
import numpy as np

# 创建一个 RealSense 管道对象
pipeline = rs.pipeline()

# 创建一个配置对象
config = rs.config()

# 配置 depth 和 color 流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    # 启动管道
    pipeline.start(config)

    # 获取设备的深度传感器
    profile = pipeline.get_active_profile()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

    # 对齐深度到颜色
    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        # 等待一组新的对齐帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # 获取对齐后的颜色和深度帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        # 确保帧是有效的
        if not aligned_depth_frame or not aligned_color_frame:
            print("无法获取对齐后的帧")
            continue

        # 获取颜色传感器的内参
        intrinsics = color_profile.get_intrinsics()

        # 构建相机矩阵
        camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy],
                                  [0, 0, 1]])

        # 获取畸变系数
        dist_coeffs = np.array(intrinsics.coeffs)

        # 打印相机矩阵和畸变系数
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
        break

finally:
    # 停止管道
    pipeline.stop()

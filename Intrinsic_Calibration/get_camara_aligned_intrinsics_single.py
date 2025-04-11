import pyrealsense2 as rs
import numpy as np
import cv2
import json

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流

try:
    pipeline.start(config)  # 流程开始

    # 获取设备的深度传感器
    profile = pipeline.get_active_profile()
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)

    while True:
        frames = pipeline.wait_for_frames()  # 等待获取图像帧

        aligned_frames = align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        # 确保帧是有效的
        if not aligned_depth_frame or not aligned_color_frame:
            print("无法获取对齐后的帧")
            continue

        aligned_color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
        aligned_depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        print(aligned_color_intrin)
        print(aligned_depth_intrin)

        # # 获取颜色传感器的内参
        # intrinsics = color_profile.get_intrinsics()

        # 获取畸变系数
        dist_coeffs = np.array(aligned_color_intrin.coeffs)


        print("Distortion Coefficients:\n", dist_coeffs)

        # 暂停以避免过多输出
        break

finally:
    # 停止管道
    pipeline.stop()

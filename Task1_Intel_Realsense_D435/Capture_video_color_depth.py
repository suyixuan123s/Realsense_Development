"""
Author: Yixuan Su
Date: 2025/02/24 12:37
File: Capture_video_color_depth.py
Description: 采集RealSense D435的RGB视频和深度图像
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from datetime import datetime
import os

# 定义保存图像和视频的目录
save_directory = r'E:\ABB-Project\ABB_wrs\suyixuan\ABB\Depth_Anything_V2\Point_cloud_Dataset\Intel_Realsense_D435170'

# 确保目录存在
os.makedirs(save_directory, exist_ok=True)

# 初始化管道
pipeline = rs.pipeline()
config = rs.config()

# 配置相机流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 启动管道
pipeline.start(config)

# 设置对齐器，将深度图与彩色图像对齐
align_to = rs.stream.color  # 对齐到颜色流
align = rs.align(align_to)

# 视频录制设置
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
video_filename = os.path.join(save_directory, f'realsense_video_{current_time}.avi')  # 改回.avi格式

# 使用更可靠的编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或 'MJPG'
frame_width = 1280
frame_height = 720
out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frame_width, frame_height))

# 确保视频写入器正常工作
if not out.isOpened():
    print("视频写入器初始化失败")
    exit()

print(f"开始录制视频到 {video_filename}")

frame_width = 1280
frame_height = 720
out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frame_width, frame_height))

# 检查视频写入器是否正常初始化
if not out.isOpened():
    print("视频写入器初始化失败，请检查编码器是否正确安装")
    exit()

# 图像保存设置
image_count = 0
image_save_interval = 1  # 每帧保存一次图像

try:
    while True:
        # 获取一帧数据
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到彩色帧
        aligned_frames = align.process(frames)

        # 获取对齐后的彩色帧和深度帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 将图像转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 如果是第一帧或者帧大小发生了变化，则更新帧大小
        if image_count == 0 or color_image.shape[1] != frame_width or color_image.shape[0] != frame_height:
            frame_width = color_image.shape[1]
            frame_height = color_image.shape[0]
            out.release()
            out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frame_width, frame_height))
            print(f"视频 writer 重新初始化，帧大小为: {frame_width}x{frame_height}")

        # 将帧写入视频
        out.write(color_image)

        # 每隔 'image_save_interval' 帧保存图像
        if image_count % image_save_interval == 0:
            # 使用datetime获取当前时间（包含毫秒）
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

            # 保存 RGB 图像
            color_image_path = os.path.join(save_directory, f'color_image_{timestamp}.jpg')
            cv2.imwrite(color_image_path, color_image)
            print(f'已保存彩色图像为 {color_image_path}')

            # 保存深度图像
            depth_image_path = os.path.join(save_directory, f'depth_image_{timestamp}.png')
            cv2.imwrite(depth_image_path, depth_image)
            print(f'已保存深度图像为 {depth_image_path}')

        # 显示彩色图像
        cv2.imshow('Color Image', color_image)

        # 按 'q' 键退出
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        image_count += 1

finally:
    # 停止管道
    pipeline.stop()

    # 释放视频 writer
    out.release()
    print(f"视频已保存到 {video_filename}")

    cv2.destroyAllWindows()

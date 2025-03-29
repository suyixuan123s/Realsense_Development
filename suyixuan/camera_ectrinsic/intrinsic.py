import pyrealsense2 as rs
import numpy as np
import cv2

# 配置RealSense相机流
pipeline = rs.pipeline()
config = rs.config()

# 启动RealSense相机流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 获取相机内参
profile = pipeline.get_active_profile()

# 获取彩色流的内参
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
color_intrinsics = color_profile.get_intrinsics()  # 彩色相机内参矩阵
print("彩色相机内参: ", color_intrinsics)

# 获取深度流的内参
depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
depth_intrinsics = depth_profile.get_intrinsics()  # 深度相机内参矩阵
print("深度相机内参: ", depth_intrinsics)

# 捕获一帧图像
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# 转换为numpy格式
color_image = np.asanyarray(color_frame.get_data())

# 停止相机
pipeline.stop()

# 保存彩色图像用于标定
cv2.imwrite("color_image.jpg", color_image)

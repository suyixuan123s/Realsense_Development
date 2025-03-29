import pyrealsense2 as rs
import numpy as np
import cv2

# 配置RealSense相机流
pipeline = rs.pipeline()
config = rs.config()

# 配置彩色流和深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动RealSense相机
pipeline.start(config)

# 获取彩色相机的内参
profile = pipeline.get_active_profile()
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_profile.get_intrinsics()

# 设置对齐方式，将深度帧对齐到彩色帧
align_to = rs.stream.color
align = rs.align(align_to)

# 假设要检测的物体在彩色图像的中心点 (320, 240)
x, y = 320, 240  # 固定检测点

try:
    while True:
        # 等待一帧图像
        frames = pipeline.wait_for_frames()

        # 对齐深度帧到彩色帧
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 对齐后的深度帧
        color_frame = aligned_frames.get_color_frame()  # 彩色帧

        # 检查是否成功获取帧
        if not aligned_depth_frame or not color_frame:
            continue

        # 将彩色帧转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 获取(x, y) = (320, 240) 点的深度信息（单位为米）
        depth_value = aligned_depth_frame.get_distance(x, y)

        # 计算中心点 (x, y) 的三维坐标 (X, Y, Z)
        fx = intrinsics.fx
        fy = intrinsics.fy
        ppx = intrinsics.ppx
        ppy = intrinsics.ppy

        # 使用公式转换为三维坐标 (X, Y, Z)
        X = (x - ppx) * depth_value / fx
        Y = (y - ppy) * depth_value / fy
        Z = depth_value

        # 在彩色图像中标记检测点 (320, 240)
        cv2.circle(color_image, (x, y), 5, (0, 255, 0), 2)  # 在 (320, 240) 绘制绿色圆圈
        cv2.putText(color_image, f"Depth: {depth_value:.2f} m", (x - 100, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示彩色图像
        cv2.imshow('Color Image with Depth', color_image)

        # 打印 (320, 240) 点的深度信息和三维坐标
        print(f"(320, 240) 点的深度: {depth_value:.2f} 米")
        print(f"(320, 240) 点的三维坐标: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f} 米")

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止相机
    pipeline.stop()
    cv2.destroyAllWindows()

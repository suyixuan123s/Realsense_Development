"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

def main():
    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始流
    pipeline.start(config)

    # 创建对齐对象，以对齐深度帧到颜色帧
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 创建 ArUco 字典和检测参数
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()

            # 对齐深度帧到颜色帧
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 转换为 numpy 数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 检测二维码
            corners, ids, _ = aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)

            if ids is not None:
                # 在图像上绘制检测到的标记
                aruco.drawDetectedMarkers(color_image, corners, ids)

                # 获取每个标记的中心点深度信息
                for corner in corners:
                    # 计算中心点
                    center = np.mean(corner[0], axis=0).astype(int)
                    depth = depth_frame.get_distance(center[0], center[1])
                    print(f"Marker center at {center} with depth {depth:.2f} meters")

            # 显示图像
            cv2.imshow('RealSense', color_image)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 停止流
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

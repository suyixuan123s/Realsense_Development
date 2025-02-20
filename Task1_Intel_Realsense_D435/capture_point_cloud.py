"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: capture_point_cloud.py
Description:

"""

import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2


def capture_point_cloud():
    # 配置RealSense相机的管道，设置深度和颜色流的格式和分辨率
    pipeline = rs.pipeline()  # 创建RealSense的管道对象，用于流数据
    config = rs.config()  # 配置对象
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 启用深度流，分辨率640x480，帧率30fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 启用彩色流，格式为BGR8，帧率30fps

    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 启用深度流，分辨率640x480，帧率30fps
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 启用彩色流，格式为BGR8，帧率30fps

    # 对齐深度和颜色帧，以便将深度信息与颜色图像进行匹配
    align_to = rs.stream.color  # 对齐到彩色流
    align = rs.align(align_to)  # 创建对齐对象，用于将深度帧与颜色帧对齐

    # 启动相机并开始捕获数据流
    pipeline.start(config)

    try:
        while True:
            # 捕获一组同步帧，并将深度帧与颜色帧对齐
            frames = pipeline.wait_for_frames()  # 等待新的一帧
            aligned_frames = align.process(frames)  # 对齐深度帧和颜色帧
            depth_frame = aligned_frames.get_depth_frame()  # 获取对齐后的深度帧
            color_frame = aligned_frames.get_color_frame()  # 获取对齐后的彩色帧

            # 如果未获取到深度或颜色帧，跳过本次循环
            if not depth_frame or not color_frame:
                print("无法获取深度或彩色帧")
                continue

            # 将彩色帧数据转换为NumPy数组，并显示图像
            color_image = np.asanyarray(color_frame.get_data())  # 将彩色帧转换为NumPy数组
            cv2.imshow('RealSense', color_image)  # 使用OpenCV显示彩色图像

            # 等待用户按下回车键以捕获点云
            key = cv2.waitKey(1)
            if key == 13:  # 如果按下回车键，开始处理点云
                # 获取深度图数据，转换为NumPy数组
                depth_image = np.asanyarray(depth_frame.get_data())  # 将深度帧转换为NumPy数组

                # 创建点云对象
                pc = rs.pointcloud()  # 创建RealSense点云对象
                pc.map_to(color_frame)  # 将彩色帧映射到点云中，用于提取纹理颜色
                points = pc.calculate(depth_frame)  # 基于深度图计算点云

                # 将点云的顶点（3D坐标）转换为Open3D格式
                vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # 获取点云顶点并转换为NumPy数组
                tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # 获取点云纹理坐标（UV）

                # 使用Open3D创建一个新的点云对象并加载顶点数据
                point_cloud = o3d.geometry.PointCloud()  # 创建Open3D点云对象
                point_cloud.points = o3d.utility.Vector3dVector(vtx)  # 设置点云的三维坐标数据

                # 修正UV坐标，确保UV坐标不超出[0, 1]的范围
                tex[:, 0] = np.clip(tex[:, 0], 0, 1)  # 限制U坐标在[0, 1]之间
                tex[:, 1] = np.clip(tex[:, 1], 0, 1)  # 限制V坐标在[0, 1]之间

                # 将UV坐标转换为图像的像素坐标
                tex_x = (tex[:, 0] * (color_image.shape[1] - 1)).astype(int)  # 将归一化的U坐标转换为像素横坐标
                tex_y = (tex[:, 1] * (color_image.shape[0] - 1)).astype(int)  # 将归一化的V坐标转换为像素纵坐标

                # 将彩色图像从BGR转换为RGB格式
                color_image_rgb = color_image[:, :, [2, 1, 0]]  # 将彩色图像的BGR格式转换为RGB格式

                # 根据UV坐标提取对应的RGB颜色信息
                colors = color_image_rgb[tex_y, tex_x, :]  # 使用像素坐标从RGB图像中提取颜色
                point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 将颜色数据归一化并赋值给点云

                # # 保存带颜色的点云到文件
                # o3d.io.write_point_cloud("../Point_cloud_Datasets/output_point_cloud.ply", point_cloud)  # 保存点云为PLY格式文件
                # print("点云文件已保存: output_point_cloud.ply")  # 输出保存成功信息
                # break  # 退出循环，停止捕获

                # 保存带颜色的点云到文件
                o3d.io.write_point_cloud(
                    "E:\ABB-Project\ABB_wrs\suyixuan\ABB\depth_anything_v2\Point_cloud_Dataset\output_point_cloud01.ply",
                    point_cloud)  # 保存点云为PLY格式文件
                print("点云文件已保存: output_point_cloud01.ply")  # 输出保存成功信息
                break  # 退出循环，停止捕获
    finally:
        # 停止相机流并关闭OpenCV窗口
        pipeline.stop()  # 停止相机捕获
        cv2.destroyAllWindows()  # 关闭OpenCV窗口


if __name__ == "__main__":
    capture_point_cloud()  # 调用主函数，开始捕获点云

"""
Author: Yixuan Su
Date: 2024/11/20 10:36
File: chessboard_center.py
Description:
"""


import numpy as np
import cv2

# 相机内参矩阵和畸变系数（使用十张图片标定得到的结果）
camera_matrix = np.array([[908.05716124, 0., 640.58062138],
                          [0., 907.14785856, 349.07025268],
                          [0., 0., 1.]])
dist_coeffs = np.array([[0.12338635, -0.09838498, -0.00406485, -0.00240096, -0.49340807]])

# 棋盘格尺寸和每个方格的大小（单位：米）
chessboard_size = (11, 8)
square_size = 0.03  # 每个格子的大小为 0.03 米（30 毫米）

# 读取新拍摄的图像
image_path = 'E:\ABB\AI\Depth-Anything-V2\suyixuan\Intel_Realsense_D435_Datasets\ABB_Intel_Realsense\Dataset2\chessboard_image_0.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"无法读取图像，请检查文件路径: {image_path}")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 准备棋盘格在世界坐标系中的坐标
        obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        obj_points *= square_size

        # 使用 solvePnP 计算相机到标定板的旋转和平移矩阵
        ret, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)

        if ret:
            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # 构造齐次变换矩阵（相机坐标系到标定板坐标系）
            transformation_camera_to_board = np.eye(4)
            transformation_camera_to_board[:3, :3] = rotation_matrix
            transformation_camera_to_board[:3, 3] = tvec[:, 0]

            # 计算标定板到相机的齐次变换矩阵
            transformation_board_to_camera = np.linalg.inv(transformation_camera_to_board)

            # 相机在标定板坐标系中的位置
            camera_position_in_board = transformation_board_to_camera[:3, 3]

            print("相机在标定板坐标系中的位置 (x, y, z):")
            print(camera_position_in_board)

            # 在图像上绘制标定板的原点
            origin = tuple(corners[0][0].astype(int))  # 获取第一个角点的像素坐标，并转换为整数
            cv2.circle(image, origin, radius=10, color=(0, 0, 255), thickness=-1)  # 在图像上绘制红色圆点表示原点

            # 定义坐标轴长度
            axis_length = 0.1  # 10 厘米

            # 计算坐标轴的终点在图像中的投影位置
            axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
            image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

            # 将坐标轴绘制到图像上
            corner = tuple(corners[0][0].astype(int))
            image_points = image_points.reshape(-1, 2)
            cv2.line(image, corner, tuple(image_points[0].astype(int)), (0, 0, 255), 5)  # X 轴 - 红色
            cv2.line(image, corner, tuple(image_points[1].astype(int)), (0, 255, 0), 5)  # Y 轴 - 绿色
            cv2.line(image, corner, tuple(image_points[2].astype(int)), (255, 0, 0), 5)  # Z 轴 - 蓝色

            # 显示带有原点和坐标轴的图像
            cv2.imshow("Chessboard with Origin and Axes", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("未能计算出相机到标定板的转换关系。")
    else:
        print("未能找到棋盘格角点，请确保标定板在视野中且清晰可见。")


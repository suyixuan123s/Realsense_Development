"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: calibrate_camera.py
Description:

"""

import numpy as np
import cv2
import os


def get_transformation_matrix(rvec, tvec):
    """
    根据旋转向量和平移向量计算齐次变换矩阵。
    :param rvec: 旋转向量
    :param tvec: 平移向量
    :return: 齐次变换矩阵
    """
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 构造齐次变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = tvec.flatten()

    return transformation_matrix


def calibrate_camera(images_dir, chessboard_size, square_size):
    """
    使用棋盘格图像标定相机，计算相机内参和每张图像的外参。
    :param images_dir: 存放棋盘格图像的目录
    :param chessboard_size: 棋盘格尺寸 (列数, 行数)
    :param square_size: 每个棋盘格的大小（以米为单位）
    :return: 相机内参矩阵、畸变系数、每张图像的齐次变换矩阵
    """
    # 准备棋盘格的世界坐标
    obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    obj_points *= square_size

    # 用于存储世界坐标和图像坐标的点
    obj_points_list = []  # 世界坐标系中的点
    img_points_list = []  # 图像平面上的点

    # 获取图像路径
    images = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith('.png')]

    for image_path in images:
        # 读取图像
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # 如果找到角点，添加世界坐标点和图像坐标点
            obj_points_list.append(obj_points)
            img_points_list.append(corners)

            # 可视化角点
            cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
            cv2.imshow('Corners', image)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # 标定相机，计算内参和畸变系数
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points_list, img_points_list,
                                                                        gray.shape[::-1], None, None)

    # 打印相机内参和畸变系数
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # 计算每张图像的齐次变换矩阵
    transformation_matrices = []
    for rvec, tvec in zip(rvecs, tvecs):
        transformation_matrix = get_transformation_matrix(rvec, tvec)
        transformation_matrices.append(transformation_matrix)
        print(f"Transformation Matrix for Image {len(transformation_matrices)}:\n", transformation_matrix)

    return camera_matrix, dist_coeffs, transformation_matrices


# 示例使用
if __name__ == '__main__':
    images_directory = r'E:\ABB-Project\cc-wrs\ABB_Intel_Realsense\Dataset'
    chessboard_size = (11, 8)  # 棋盘格尺寸 (列数, 行数)
    square_size = 0.03  # 每个棋盘格的大小（单位：米）

    calibrate_camera(images_directory, chessboard_size, square_size)

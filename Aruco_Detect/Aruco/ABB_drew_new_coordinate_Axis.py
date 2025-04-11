"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

# 设置相机内参和畸变系数
camera_matrix = np.array([[434.43981934, 0, 322.23464966],
                          [0, 433.24884033, 236.84153748],
                          [0, 0, 1]])

dist_coeffs = np.array([-0.05277087, 0.06000207, 0.00087849, 0.00136543, -0.01997724])

# 设置相机内参和畸变系数
fx = 434.43981934
fy = 433.24884033
cx = 322.23464966
cy = 236.84153748

# 设置相机内参和畸变系数
depth_scale = 9.999999747378752e-05

# 相机内参
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# 创建 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
# 创建检测参数
parameters = aruco.DetectorParameters()
# 标记的实际大小 (米)
marker_size = 0.025


def detect_aruco_and_transform_coordinates(image, camera_matrix, dist_coeffs, aruco_dict, aruco_params):
    """
    检测 ArUco 标记，按照 ID 顺序连接中心点，并转换坐标系

    Args
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        aruco_dict: ArUco 字典
        aruco_params: ArUco 参数

    Returns:
        marker_centers_transformed: 转换后的坐标
        image: 带有连接线的图像
    """

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 畸变校正
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(gray, camera_matrix, dist_coeffs, None, newcameramtx) # 对图像进行去畸变处理

    # 检测 ArUco 标记
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) == 0:
        print("未检测到 ArUco 标记。")
        return None, image

    # 创建一个字典来存储检测到的标记的中心点，按照 ID 排序
    marker_centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        marker_centers[marker_id] = np.mean(corners[i][0], axis=0)


    # 按照 ID 排序
    sorted_ids = np.sort(ids.flatten())

    # 定义新的坐标系
    if len(sorted_ids) >= 2:
        origin_id = sorted_ids[0]
        x_axis_id = sorted_ids[1]

        origin = marker_centers[origin_id]
        x_axis_point = marker_centers[x_axis_id]

        # 计算 X 轴向量
        x_axis_vector = x_axis_point - origin

        # 计算旋转角度 (弧度)
        angle = np.arctan2(x_axis_vector[1], x_axis_vector[0])

        # 创建旋转矩阵
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        # 转换所有中心点到新的坐标系
        marker_centers_transformed = {}
        for marker_id, center in marker_centers.items():
            # 平移到原点
            translated_center = center - origin
            # 旋转
            transformed_center = np.dot(rotation_matrix, translated_center)
            marker_centers_transformed[marker_id] = transformed_center

    else:
        print("需要至少两个 ArUco 标记才能定义坐标系。")
        return None, image

    # 连接中心点
    if len(sorted_ids) > 1:
        for i in range(len(sorted_ids) - 1):
            id1 = sorted_ids[i]
            id2 = sorted_ids[i + 1]
            center1 = marker_centers[id1]
            center2 = marker_centers[id2]
            # tuple() 将 center 的坐标转换为整数元组，以便于绘制
            cv2.line(image, tuple(center1.astype(int)), tuple(center2.astype(int)), (0, 255, 0), 2)  # 绿色线

    # 可视化检测到的标记
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    return marker_centers_transformed, image


if __name__ == '__main__':
    # 读取图像
    image = cv2.imread(
        r'../../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/color_image_20250411-102019.jpg')  # 替换为你的图像路径

    # 检测 ArUco 标记并转换坐标系
    marker_centers_transformed, image_with_lines = detect_aruco_and_transform_coordinates(image, camera_matrix,
                                                                                          dist_coeffs, aruco_dict,
                                                                                          parameters)

    if marker_centers_transformed:
        # 使用 matplotlib 显示结果
        plt.figure(figsize=(12, 6))

        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
        plt.title("Original Image with Connected Centers")

        # 显示转换后的坐标
        plt.subplot(1, 2, 2)
        for marker_id, center in marker_centers_transformed.items():
            plt.plot(center[0], center[1], 'ro')  # 红色点
            plt.text(center[0], center[1], str(marker_id))  # 显示 ID

        # 设置坐标轴标签和标题
        plt.xlabel("X (Transformed)")
        plt.ylabel("Y (Transformed)")
        plt.title("Transformed Marker Centers")
        # 设置图形属性
        plt.grid(True)  # 启用网格线，以便于观察数据点的位置
        plt.axis('equal')  # 保持 X 和 Y 轴的比例一致

        # 调整布局并显示图形
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域，避免重叠

        # 保存子图
        plt.subplot(1, 2, 1)  # 选择第一个子图
        plt.savefig('original_image_with_connected_centers.png')  # 保存原始图像子图

        plt.subplot(1, 2, 2)  # 选择第二个子图
        plt.savefig('transformed_marker_centers.png')  # 保存转换后的坐标子图

        # 保存整个图形
        plt.savefig('combined_results.png')  # 保存整个图形

        plt.show()
    else:
        print("未检测到足够的 ArUco 标记，无法显示结果。")

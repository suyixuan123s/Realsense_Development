"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import cv2
import cv2.aruco as aruco
import numpy as np

# 设置相机内参和畸变系数（需要根据您的相机校准结果设置）
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

def detect_aruco_and_connect_centers(image, camera_matrix, dist_coeffs, aruco_dict, aruco_params):
    """
    检测 ArUco 标记并按照 ID 顺序连接中心点。

    Args:
        image: 输入图像。
        camera_matrix: 相机内参矩阵。
        dist_coeffs: 畸变系数。
        aruco_dict: ArUco 字典。
        aruco_params: ArUco 参数。

    Returns:
        None (直接在图像上绘制连接线)。
    """

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 畸变校正
    h,  w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    dst = cv2.undistort(gray, camera_matrix, dist_coeffs, None, newcameramtx)

    # 检测 ArUco 标记
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=aruco_params)

    if ids is None or len(ids) == 0:
        print("未检测到 ArUco 标记。")
        return

    # 创建一个字典来存储检测到的标记的中心点，按照 ID 排序
    marker_centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        marker_centers[marker_id] = np.mean(corners[i][0], axis=0)

    # 按照 ID 排序
    sorted_ids = np.sort(ids.flatten())

    # 连接中心点
    if len(sorted_ids) > 1:
        for i in range(len(sorted_ids) - 1):
            id1 = sorted_ids[i]
            id2 = sorted_ids[i+1]
            center1 = marker_centers[id1]
            center2 = marker_centers[id2]
            cv2.line(image, tuple(center1.astype(int)), tuple(center2.astype(int)), (0, 255, 0), 2)  # 绿色线

    # 可视化检测到的标记
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

# 主程序
if __name__ == '__main__':
    # 读取图像
    image = cv2.imread(r'../../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/color_image_20250410-232856.jpg')  # 替换为你的图像路径

    # 检测 ArUco 标记并连接中心点
    detect_aruco_and_connect_centers(image, camera_matrix, dist_coeffs, aruco_dict, parameters)

    # 显示图像
    cv2.imshow("ArUco Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

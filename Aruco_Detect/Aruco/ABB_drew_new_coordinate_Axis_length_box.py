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
        origin: ArUco[0]在图像坐标系中的位置
    """

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 畸变校正
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(gray, camera_matrix, dist_coeffs, None, newcameramtx)  # 对图像进行去畸变处理

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
    if len(sorted_ids) >= 3:
        origin_id = sorted_ids[0]
        x_axis_id = sorted_ids[1]
        y_axis_id = sorted_ids[2]

        origin = marker_centers[origin_id]
        x_axis_point = marker_centers[x_axis_id]
        y_axis_point = marker_centers[y_axis_id]
        # 计算 X 轴和 Y 轴的长度
        x_axis_length = np.linalg.norm(x_axis_point - origin)
        y_axis_length = np.linalg.norm(y_axis_point - origin)
        print(f"X 轴长度（像素距离）: {x_axis_length}")
        print(f"Y 轴长度（像素距离）: {y_axis_length}")

        # 计算 X 轴向量
        x_axis_vector = x_axis_point - origin

        # 计算 Y 轴向量
        y_axis_vector = y_axis_point - origin

        # 创建旋转矩阵 (假设 X 轴和 Y 轴垂直)
        # 归一化 X 轴向量
        x_axis_unit_vector = x_axis_vector / np.linalg.norm(x_axis_vector)

        # 计算 Y 轴的单位向量，使其与 X 轴垂直
        y_axis_unit_vector = np.array([-x_axis_unit_vector[1], x_axis_unit_vector[0]])

        rotation_matrix = np.array([x_axis_unit_vector, y_axis_unit_vector]).T  # 转置

        # 转换所有中心点到新的坐标系
        marker_centers_transformed = {}
        sorted_marker_ids = sorted(marker_centers.keys())  # 对 marker_centers 的键进行排序
        for marker_id in sorted_marker_ids:
            center = marker_centers[marker_id]

            # 平移到原点
            translated_center = center - origin

            # # 旋转
            # transformed_center = np.dot(rotation_matrix, translated_center)

            # 计算在 X 轴和 Y 轴上的投影长度
            x_coordinate = np.dot(translated_center, x_axis_unit_vector)
            y_coordinate = np.dot(translated_center, y_axis_unit_vector)

            print("x_coordinate, y_coordinate", x_coordinate, y_coordinate)

            marker_centers_transformed[marker_id] = np.array([x_coordinate, y_coordinate])

    else:
        print("需要至少三个 ArUco 标记才能定义坐标系。")
        return None, image

    # 连接中心点
    if len(sorted_ids) > 1:
        origin_id = sorted_ids[0]  # 获取原点的 ID
        origin_center = marker_centers[origin_id]  # 获取原点的中心点

        for i in range(1, len(sorted_ids)):  # 从 1 开始，跳过原点
            id = sorted_ids[i]
            center = marker_centers[id]
            # tuple() 将 center 的坐标转换为整数元组，以便于绘制
            cv2.line(image, tuple(origin_center.astype(int)), tuple(center.astype(int)), (0, 255, 0), 2)  # 绿色线

    # 可视化检测到的标记
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    return marker_centers_transformed, image, origin


def detect_nut(image):
    """

    检测图像中的螺母，返回其中心点坐标

    零阶矩 (M00)：
    M00 = Σ Σ f(x, y)
    表示图像中所有像素强度值的总和，对于二值图像来说，就是图像中白色区域的面积

    一阶矩 (M10, M01)：
    M10 = Σ Σ x * f(x, y)
    M01 = Σ Σ y * f(x, y)
    用于计算图像的质心（中心点）坐标

    二阶矩 (M20, M02, M11)：
    M20 = Σ Σ x^2 * f(x, y)
    M02 = Σ Σ y^2 * f(x, y)
    M11 = Σ Σ x * y * f(x, y)
    用于描述图像的形状和方向

    中心矩 (μpq)：
    中心矩是对平移具有不变性的矩，通过将坐标原点移动到图像的质心来实现
    计算公式比较复杂，但其目的是消除图像平移对矩的影响

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # 在边缘图像中查找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个空列表，用于存储检测到的螺母中心点坐标
    nut_centers = []
    # 遍历所有检测到的轮廓
    for contour in contours:
        # 计算轮廓的周长
        peri = cv2.arcLength(contour, True)
        # 使用指定精度逼近多边形曲线
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # 螺母通常具有六边形或圆形形状
        # 判断逼近后的多边形的顶点数是否大于等于 6，至少6个顶点
        if len(approx) >= 6:  # 至少6个顶点
            # 计算轮廓的面积和中心点
            area = cv2.contourArea(contour)
            # 如果轮廓的面积大于 100，则认为是一个有效的螺母轮廓，面积阈值，去除小噪点
            if area > 100:  # 面积阈值，去除小噪点
                # 计算轮廓的矩
                M = cv2.moments(contour)
                # 如果轮廓的零阶矩不为 0，则计算轮廓的中心点坐标
                if M["m00"] != 0:
                    # 计算轮廓的中心点 X 坐标
                    cX = int(M["m10"] / M["m00"])
                    # 计算轮廓的中心点 Y 坐标
                    cY = int(M["m01"] / M["m00"])
                    # 将计算得到的中心点坐标添加到 nut_centers 列表中
                    nut_centers.append((cX, cY))

    return nut_centers


def process_image(image, camera_matrix, dist_coeffs, aruco_dict, aruco_params):
    """
    检测 ArUco 标记，检测螺母，并转换坐标系

    Args:
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        aruco_dict: ArUco 字典
        aruco_params: ArUco 检测参数

    Returns:
        image: 处理后的图像
        nut_positions: 螺母在新坐标系中的坐标列表
    """
    # 1. ArUco 标记检测和坐标转换
    marker_centers_transformed, image, origin = detect_aruco_and_transform_coordinates(image, camera_matrix,
                                                                                       dist_coeffs,
                                                                                       aruco_dict, aruco_params)

    nut_positions = []

    if marker_centers_transformed is not None:
        # 2. 螺母检测
        nut_centers = detect_nut(image)
        if nut_centers:
            for nut_center in nut_centers:
                # 3. 转换螺母坐标到新坐标系
                nut_center = np.array(nut_center, dtype=np.float32)

                # 计算螺母相对于 ArUco 坐标系原点的偏移量
                translated_center = nut_center - origin

                # 直接使用螺母相对于 ArUco 坐标系原点的偏移量作为新坐标系中的坐标
                nut_position_new_coordinate = translated_center

                nut_positions.append(nut_position_new_coordinate)

                # 4. 在图像上显示螺母位置
                cv2.circle(image, tuple(nut_center.astype(int)), 5, (0, 255, 0), -1)  # 绿色圆点
                cv2.putText(image, "Nut", tuple((nut_center + np.array([10, -10])).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 在图像上显示新坐标系中的坐标
                text_position = tuple((nut_center + np.array([10, 20])).astype(int))
                cv2.putText(image, f"({nut_position_new_coordinate[0]:.2f}, {nut_position_new_coordinate[1]:.2f})",
                            text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, nut_positions, marker_centers_transformed

def process_image_NewXY (image, camera_matrix, dist_coeffs, aruco_dict, aruco_params):
    """
    检测 ArUco 标记，检测螺母，并转换坐标系

    Args:
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        aruco_dict: ArUco 字典
        aruco_params: ArUco 检测参数

    Returns:
        image: 处理后的图像
        nut_positions: 螺母在新坐标系中的坐标列表
        marker_centers_transformed: 转换后的 ArUco 标记中心点坐标字典
    """
    # 1. ArUco 标记检测和坐标转换
    marker_centers_transformed, image, origin = detect_aruco_and_transform_coordinates(image, camera_matrix,
                                                                                       dist_coeffs,
                                                                                       aruco_dict, aruco_params)

    nut_positions = []

    if marker_centers_transformed is not None:
        # 2. 螺母检测
        nut_centers = detect_nut(image)
        if nut_centers:
            # 确定 XY 轴的范围
            x_min = min(marker_centers_transformed[marker_id][0] for marker_id in marker_centers_transformed)
            x_max = max(marker_centers_transformed[marker_id][0] for marker_id in marker_centers_transformed)
            y_min = min(marker_centers_transformed[marker_id][1] for marker_id in marker_centers_transformed)
            y_max = max(marker_centers_transformed[marker_id][1] for marker_id in marker_centers_transformed)

            for nut_center in nut_centers:
                # 3. 转换螺母坐标到新坐标系
                nut_center = np.array(nut_center, dtype=np.float32)

                # 计算螺母相对于 ArUco 坐标系原点的偏移量
                translated_center = nut_center - origin

                # 直接使用螺母相对于 ArUco 坐标系原点的偏移量作为新坐标系中的坐标
                nut_position_new_coordinate = translated_center

                # 4. 判断螺母是否在 XY 轴范围内
                if x_min <= nut_position_new_coordinate[0] <= x_max and \
                   y_min <= nut_position_new_coordinate[1] <= y_max:
                    nut_positions.append(nut_position_new_coordinate)

                    # 5. 在图像上显示螺母位置
                    cv2.circle(image, tuple(nut_center.astype(int)), 5, (0, 255, 0), -1)  # 绿色圆点
                    cv2.putText(image, "Nut", tuple((nut_center + np.array([10, -10])).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 在图像上显示新坐标系中的坐标
                    text_position = tuple((nut_center + np.array([10, 20])).astype(int))
                    cv2.putText(image, f"({nut_position_new_coordinate[0]:.2f}, {nut_position_new_coordinate[1]:.2f})",
                                text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, nut_positions, marker_centers_transformed



if __name__ == "__main__":
    # 初始化相机参数和 ArUco 检测器
    camera_matrix = np.array([[434.43981934, 0.0, 318.67144775],
                              [0.0, 434.35751343, 241.73374939],
                              [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([[-0.04251478, 0.12392275, 0.00148438, 0.00089415, -0.23152345]])

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    aruco_params = aruco.DetectorParameters()

    # 读取图像
    image = cv2.imread(
        "../../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/color_image_20250411-113817.jpg")

    if image is None:
        print("无法读取图像，请检查路径是否正确。")
    else:
        # 处理图像
        # processed_image, nut_positions, marker_centers_transformed = process_image(image, camera_matrix, dist_coeffs,
        #                                                                            aruco_dict, aruco_params)

        processed_image, nut_positions, marker_centers_transformed = process_image_NewXY(image, camera_matrix, dist_coeffs,
                                                                                   aruco_dict, aruco_params)



        # 使用 matplotlib 显示结果
        if marker_centers_transformed:
            plt.figure(figsize=(12, 6))

            # 显示原始图像
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image with Connected Centers and Nut Detections")

            # 显示转换后的坐标
            plt.subplot(1, 2, 2)
            for marker_id, center in marker_centers_transformed.items():
                plt.plot(center[0], center[1], 'ro')  # 红色点
                plt.text(center[0], center[1], str(marker_id))  # 显示 ID

            # 添加螺母
            for nut_position in nut_positions:
                plt.plot(nut_position[0], nut_position[1], 'go')  # 绿色点
                plt.text(nut_position[0], nut_position[1], "Nut")  # 显示 "Nut" 文本

            # 设置坐标轴标签和标题
            plt.xlabel("X (Transformed)")
            plt.ylabel("Y (Transformed)")
            plt.title("Transformed Marker Centers and Nut Detections")
            # 设置图形属性
            plt.grid(True)  # 启用网格线，以便于观察数据点的位置
            plt.axis('equal')  # 保持 X 和 Y 轴的比例一致

            # 调整布局并显示图形
            plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域，避免重叠

            # 保存子图
            plt.savefig('original_image_with_connected_centers.png')  # 保存原始图像子图

            # plt.savefig('original_image_with_connected_centers.png', bbox_inches='tight')  # 保存原始图像子图
            # plt.savefig('transformed_marker_centers.png', bbox_inches='tight')  # 保存转换后的坐标子图
            #
            # # 保存整个图形
            # plt.savefig('combined_results.png', bbox_inches='tight')  # 保存整个图形

            plt.show()




        else:
            print("未检测到足够的 ArUco 标记，无法显示结果。")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 打印螺母位置
        if nut_positions:
            print("检测到的螺母位置（新坐标系）:")
            for position in nut_positions:
                print(f"({position[0]:.2f}, {position[1]:.2f})")
        else:
            print("未检测到螺母。")

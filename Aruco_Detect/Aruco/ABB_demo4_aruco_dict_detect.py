"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import cv2
import numpy as np
import cv2.aruco as aruco


def drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, length):
    """
    绘制 3D 坐标轴到图像上.
    Args:
        img: 要绘制坐标轴的图像.
        camera_matrix: 相机内参矩阵.
        dist_coeffs: 畸变系数.
        rvec: 旋转向量.
        tvec: 平移向量.
        length: 坐标轴的长度.
    """
    axes_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]).reshape(-1, 3)
    imgpts, jac = cv2.projectPoints(axes_points, rvec, tvec, camera_matrix, dist_coeffs)

    imgpts = np.int32(imgpts).reshape(-1, 2)

    # 绘制坐标轴
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X轴，蓝色
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y轴，绿色
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z轴，红色

    return img


def fitPlane(points):
    """
    使用最小二乘法拟合平面.
    Args:
        points: 一个 Nx3 的 NumPy 数组，表示 N 个 3D 点.

    Returns:
        plane_normal: 平面的法向量.
        plane_d: 平面方程的常数项.
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, vh = np.linalg.svd(points_centered)
    plane_normal = vh[-1, :]  # 法向量是最后一个奇异向量
    plane_d = -np.dot(plane_normal, centroid)
    return plane_normal, plane_d


def visualize_plane_in_image(image, points_3d, camera_matrix, dist_coeffs, color=(0, 255, 0), thickness=2):
    """
    在二维图像上可视化由四个三维点定义的平面区域。

    Args:
        image: 要绘制的二维图像 (NumPy 数组)。
        points_3d: 四个三维点的 NumPy 数组，形状为 (4, 3)。  这些点定义了平面区域的四个角点
        camera_matrix: 相机内参矩阵。
        dist_coeffs: 相机畸变系数。
        color: 绘制线条的颜色 (BGR 格式)。
        thickness: 绘制线条的粗细。
    """

    # 将三维点投影到二维图像上
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2).astype(int)

    # 绘制连接四个点的线条，形成平面区域的边界
    for i in range(4):
        cv2.line(image, tuple(points_2d[i]), tuple(points_2d[(i + 1) % 4]), color, thickness)

    return image


def RodriguesToRotationMatrix(rvec):
    """
    将 Rodrigues 向量转换为旋转矩阵
    """
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    return rotation_matrix


def plane_to_camera_transform(plane_normal, points_3d):
    """
    计算平面到相机坐标系的转换矩阵
    Args:
        plane_normal: 平面法向量 (NumPy 数组，形状为 (3,))
        points_3d: 四个二维码中心点的三维坐标 (NumPy 数组，形状为 (4, 3))

    Returns:
        R_plane_to_camera: 旋转矩阵 (NumPy 数组，形状为 (3, 3))
        t_plane_to_camera: 平移向量 (NumPy 数组，形状为 (3,))
    """

    # 1. 计算平面上的一点
    point_on_plane = np.mean(points_3d, axis=0)

    # 2. 确保法向量已归一化
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # 定义相机的 z 轴
    z_axis = np.array([0, 0, 1])

    # 计算旋转向量，将平面法向量对齐到相机的 z 轴
    rotation_vector = np.cross(plane_normal, z_axis)

    # 检查旋转向量的模是否接近零
    if np.linalg.norm(rotation_vector) < 1e-6:
        # 如果旋转向量接近零，使用单位矩阵作为旋转矩阵
        R_plane_to_camera = np.eye(3)
    else:
        # 计算旋转角度，确保输入值在 [-1, 1] 范围内以避免数值不稳定
        rotation_angle = np.arccos(np.clip(np.dot(plane_normal, z_axis), -1.0, 1.0))
        # 归一化旋转向量并乘以旋转角度
        rotation_vector = rotation_vector / np.linalg.norm(rotation_vector) * rotation_angle

    R_plane_to_camera = RodriguesToRotationMatrix(rotation_vector)

    # 平移向量设置为平面上的一个点
    t_plane_to_camera = R_plane_to_camera @ point_on_plane

    return R_plane_to_camera, t_plane_to_camera


def transform_point_to_image(point_object, average_transformation_matrix, K):
    """
    将物体坐标系中的点转换为图像坐标系中的像素位置

    :param point_object: 物体坐标系中的点 [x, y, z]
    :param average_transformation_matrix: 物体到相机坐标系的转换矩阵 (4x4)
    :param K: 相机内参矩阵 (3x3)
    :return: 图像坐标系中的像素位置 [u', v']
    """
    # 将物体坐标系中的点转换为齐次坐标
    point_object_homogeneous = np.array([point_object[0], point_object[1], point_object[2], 1.0])

    # 使用转换矩阵将点转换到相机坐标系
    point_camera_homogeneous = np.dot(average_transformation_matrix, point_object_homogeneous)

    # 提取相机坐标系中的点
    X, Y, Z, _ = point_camera_homogeneous

    # 使用相机内参矩阵将相机坐标系中的点投影到图像平面
    point_image_homogeneous = np.dot(K, np.array([X, Y, Z]))

    # 归一化以获得像素坐标
    u = point_image_homogeneous[0] / point_image_homogeneous[2]
    v = point_image_homogeneous[1] / point_image_homogeneous[2]

    return np.array([u, v])


def pixel_to_camera_coordinates(pixel_position, K, Z=1.0):
    """
    将图像坐标系中的像素位置转换为相机坐标系中的位置

    :param pixel_position: 图像坐标系中的像素位置 [u, v]
    :param K: 相机内参矩阵 (3x3)
    :param Z: 假设的深度值
    :return: 相机坐标系中的位置 [X, Y, Z]
    """
    # 计算相机内参矩阵的逆
    K_inv = np.linalg.inv(K)
    # 将像素坐标转换为齐次坐标
    pixel_homogeneous = np.array([pixel_position[0], pixel_position[1], 1.0])
    # 使用逆矩阵将像素坐标转换为归一化相机坐标
    camera_normalized = np.dot(K_inv, pixel_homogeneous)
    # 计算相机坐标系中的位置
    X = camera_normalized[0] * Z
    Y = camera_normalized[1] * Z
    return np.array([X, Y, Z])


def camera_to_pixel_coordinates(point_camera, K):
    """
    将相机坐标系中的点转换为图像坐标系中的像素位置

    :param point_camera: 相机坐标系中的点 [X, Y, Z]
    :param K: 相机内参矩阵 (3x3)
    :return: 图像坐标系中的像素位置 [u, v]
    """
    # 将相机坐标系中的点转换为齐次坐标
    point_camera_homogeneous = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])

    # 使用相机内参矩阵进行投影
    uvw = np.dot(K, point_camera_homogeneous[:3])

    # 归一化以获得像素坐标
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]

    return np.array([u, v])


# 设置相机内参和畸变系数（需要根据您的相机校准结果设置）
camera_matrix = np.array([[434.43981934, 0, 322.23464966],
                          [0, 433.24884033, 236.84153748],
                          [0, 0, 1]])

k1 = -0.05277087
k2 = 0.06000207
p1 = 0.00087849
p2 = 0.00136543
k3 = -0.01997724

# 设置相机内参和畸变系数（需要根据您的相机校准结果设置）
fx = 434.43981934
fy = 433.24884033
cx = 322.23464966
cy = 236.84153748

# 假设您有相机内参
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

dist_coeffs = np.array([k1, k2, p1, p2, k3])

# 创建 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# 创建检测参数
parameters = aruco.DetectorParameters()

# 假设标记的实际大小为 100 毫米（即 0.1 米）
marker_size = 0.025

# 读取图像
image = cv2.imread(r'../../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/color_image_20250410-140335.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测标记
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)


if ids is not None:
    # 估计姿态 返回旋转向量 rvecs 和平移向量 tvecs
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

    print("tvecs shape:", tvecs.shape)
    print()

    # 获取所有标记的 3D 坐标
    points_3d = np.array([tvecs[i].flatten() for i in range(len(ids))])
    # depth = np.mean(tvecs[i][:, 2])  # 取所有二维码的平均深度

    print("points_3d", points_3d)
    print()

    # ------------------使用 三维坐标直接计算 距离 -------------
    # 获取 x 和 y 坐标
    x_coords = points_3d[:, 0]
    y_coords = points_3d[:, 1]

    # 计算最小和最大值
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # 计算真实的宽度和高度
    real_width = x_max - x_min
    real_height = y_max - y_min

    print("真实包围框的宽度:", real_width)
    print("真实包围框的高度:", real_height)
    print()

    # ------------------------------------

    # 投影到二维平面
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2)  # 将结果转换为 (N, 2) 的形状

    # 计算包围框
    x_min = np.min(points_2d[:, 0])
    x_max = np.max(points_2d[:, 0])
    y_min = np.min(points_2d[:, 1])
    y_max = np.max(points_2d[:, 1])

    # 确保 tvecs 的形状正确
    tvecs = tvecs.squeeze()  # 变为 (4, 3)

    # -----------------  使用三维投影二维计算距离信息 --------------

    # 计算包围框，并记录索引
    x_min_index = np.argmin(points_2d[:, 0])
    x_max_index = np.argmax(points_2d[:, 0])
    y_min_index = np.argmin(points_2d[:, 1])
    y_max_index = np.argmax(points_2d[:, 1])

    # x_min = points_2d[x_min_index, 0]
    # x_max = points_2d[x_max_index, 0]
    # y_min = points_2d[y_min_index, 1]
    # y_max = points_2d[y_max_index, 1]

    # 包围框的长宽
    pixel_width = x_max - x_min
    pixel_height = y_max - y_min

    # 获取深度信息 (修改这里)
    depth_min = tvecs[x_min_index, 2]  # 对应于 x_min 的深度
    depth_max = tvecs[x_max_index, 2]  # 对应于 x_max 的深度
    depth_ymin = tvecs[y_min_index, 2]  # 对应于 y_min 的深度
    depth_ymax = tvecs[y_max_index, 2]  # 对应于 y_max 的深度

    real_width = (pixel_width / camera_matrix[0, 0]) * depth_min  # 使用 x_min 的深度
    real_height = (pixel_height / camera_matrix[1, 1]) * depth_ymin  # 使用 y_min 的深度

    print("包围框的宽度:", pixel_width)
    print("包围框的高度:", pixel_height)
    print("实际包围框的宽度:", real_width)
    print("实际包围框的高度:", real_height)

    # 打印对应的二维码 ID
    print("x_min 对应的二维码 ID:", ids[x_min_index])
    print("x_max 对应的二维码 ID:", ids[x_max_index])
    print("y_min 对应的二维码 ID:", ids[y_min_index])
    print("y_max 对应的二维码 ID:", ids[y_max_index])
    print()

    # ===================  使用三维投影二维计算距离信息 ======================

    # ========================= 绘制中心点 =======================================================================

    # 计算四个角点的平均值，作为盒子的中心点
    box_center = np.mean(points_3d, axis=0)
    print("Box Center:", box_center)
    # 将中心点投影到图像上
    # 使用 cv2.projectPoints 将 3D 中心点投影到 2D 图像平面
    center_2d, _ = cv2.projectPoints(box_center.reshape(1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    print("center_2d", center_2d)
    # 将投影的中心点转换为整数坐标
    center_2d = center_2d[0][0].astype(int)
    print("center_2d", center_2d)
    # 在图像上绘制中心点
    cv2.circle(image, (center_2d[0], center_2d[1]), radius=10, color=(0, 0, 255), thickness=-1)  # 红色圆点

    #  =========================================================================================================

    # 使用最小二乘法拟合平面
    plane_normal, plane_d = fitPlane(points_3d)
    # 打印平面方程
    print("Plane Equation: %.2fx + %.2fy + %.2fz + %.2f = 0" % (
        plane_normal[0], plane_normal[1], plane_normal[2], plane_d))
    print()
    print("Plane Normal:", plane_normal)
    print()
    print("Plane D:", plane_d)
    print()

    # 计算转换矩阵
    R_plane_to_camera, t_plane_to_camera = plane_to_camera_transform(plane_normal, points_3d)

    # 打印结果
    print("旋转矩阵 (Plane to Camera):\n", R_plane_to_camera)
    print("平移向量 (Plane to Camera):\n", t_plane_to_camera)
    print()

    # 定义平面上的任意一点
    point_on_plane = np.array([0, 0, 0])  # 替换为实际的平面坐标

    point_in_camera = R_plane_to_camera @ point_on_plane + t_plane_to_camera
    print("平面上的点转换为相机坐标系下的坐标为:", point_in_camera)
    print()

    # ----------------  计算像素坐标到相机坐标的转换 --------
    # 设置相机内参和畸变系数
    fx = 434.43981934
    fy = 433.24884033
    cx = 322.23464966
    cy = 236.84153748
    depth_scale = 9.999999747378752e-05
    # 物体在图像中的像素坐标
    u, v = 327, 219  # 替换为实际的像素坐标
    # u, v = 327.18533004, 218.89644287
    # 将像素坐标转换为归一化坐标
    # x_norm = (u - cx) / fx
    #
    # y_norm = (v - cy) / fy
    #
    # print("x_norm", x_norm)
    # print("y_norm", y_norm)
    x_norm = 0.011395549301905743
    y_norm = -0.04141983299096996

    # ------------- 深度值的计算 -----------
    depth_map = cv2.imread('../../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/depth_image_20250410-140335.png',
                           cv2.IMREAD_UNCHANGED)  # 读取深度图
    # 获取该像素的深度值
    # depth_value = depth_map[v, u] * depth_scale
    depth_value = 0.26910057
    print(f"Depth value at pixel (u, v)({u}, {v}) = {depth_value}\n")
    # 使用深度信息计算相机坐标系下的三维坐标
    X = x_norm * depth_value
    Y = y_norm * depth_value
    Z = depth_value
    camera_coords = np.array([X, Y, Z])
    print("相机坐标系下的三维坐标:", camera_coords)

    P_object_plane = np.array([x_norm * depth_value, y_norm * depth_value, depth_value])
    print("相机坐标系下的三维坐标P_object_plane:", P_object_plane)
    print()

    # 将物体的位置转换到相机坐标系
    P_object_camera = R_plane_to_camera @ P_object_plane + t_plane_to_camera
    print("Object Position in Camera Coordinate System:", P_object_camera)
    # -----------------------------------------------------------------------------

    # ---------------------------  计算设定长宽的显示问题--------------------------
    # # 2.定义矩形区域的宽度和高度
    # width = 0.15
    # height = 0.15
    #
    # # 定义边界点 (示例：矩形的四个角点)
    # boundary_points_2d = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    #
    # # 将边界点投影到平面上
    # boundary_points_3d = []
    # for point_2d in boundary_points_2d:
    #     # 假设边界点的 z 坐标为 0
    #     point_3d = np.append(point_2d, 0)
    #     point_3d_demo = np.array(point_3d)
    #
    #     # 计算点到平面的距离
    #     distance = np.dot(plane_normal, point_3d) + plane_d
    #
    #     # 将点投影到平面上
    #     projected_point = point_3d - distance * plane_normal
    #
    #     boundary_points_3d.append(projected_point)
    #
    # boundary_points_3d = np.array(boundary_points_3d)
    #
    # # 计算平面的宽度和高度
    # width = np.linalg.norm(boundary_points_3d[1] - boundary_points_3d[0])
    # height = np.linalg.norm(boundary_points_3d[3] - boundary_points_3d[0])
    #
    # print("Plane width:", width)
    # print("Plane height:", height)

    # # 可视化平面区域
    # image = visualize_plane_in_image(image, boundary_points_3d, camera_matrix, dist_coeffs)

    # ---------------------------  计算设定长宽的显示问题--------------------------
    # 可视化Aurco的中心点的平面区域
    image = visualize_plane_in_image(image, points_3d, camera_matrix, dist_coeffs)

    # ---------------计算平均值 --------------------

    sum_rotation_matrix = np.zeros((3, 3), dtype=np.float64)
    sum_translation_vector = np.zeros((3,), dtype=np.float64)

    # 计算每个Aurco的旋转和平移信息
    for i in range(len(ids)):
        rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
        tvec = tvecs[i].reshape(1, 3)

        # 输出标记的旋转和平移向量
        print(f"Marker ID: {ids[i][0]}")
        print(f"Rotation Vector: {rvecs[i].flatten()}")
        print(f"Translation Vector: {tvecs[i].flatten()}")

        transformation_matrix = np.zeros((4, 4), dtype=np.float64)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = tvec.flatten()
        transformation_matrix[3, 3] = 1.0
        print(f"Transformation Matrix for Marker to Camera {ids[i][0]}:", transformation_matrix)

        P_object_plane1 = np.array([0, 0, 0])
        Transformation_Matrix = np.array(transformation_matrix)
        P_object_plane_homogeneous2 = np.append(P_object_plane1, 1)

        P_object_camera22 = Transformation_Matrix @ P_object_plane_homogeneous2

        P_object_camera2_3D2 = P_object_camera22[:3]
        np.set_printoptions(suppress=True)
        print(f"{ids[i][0]}:", P_object_camera2_3D2)

        pixel_position = camera_to_pixel_coordinates(P_object_plane, K)
        print("图像坐标系中的像素位置:", pixel_position)

        point_in_camera = R_plane_to_camera @ point_on_plane + t_plane_to_camera
        print("平面上的点转换为相机坐标系下的坐标为:", point_in_camera)
        print()

        # pixel_position = transform_point_to_image(P_object_plane, Transformation_Matrix, K)
        # print(f"{ids[i][0]}:像素位置:", pixel_position)

        # 累加旋转矩阵和平移向量
        sum_rotation_matrix += rotation_matrix
        sum_translation_vector += tvec.flatten()

        # 绘制坐标轴
        image = drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)

        # # 3. 像素坐标到相机坐标
        # fx = camera_matrix[0, 0]
        # fy = camera_matrix[1, 1]
        # cx = camera_matrix[0, 2]
        # cy = camera_matrix[1, 2]
        #
        # # 计算射线方向
        # x = (pixel_x - cx) / fx
        # y = (pixel_y - cy) / fy
        # direction = np.array([x, y, 1])
        #
        # # 4. 射线与平面求交
        # # 求解 t
        # t = -plane_d / np.dot(np.array(plane_normal[0], plane_normal[1], plane_normal[2]), direction)
        #
        # # 计算交点坐标
        # point_3d = t * direction
        #
        # print("像素点 ({}, {}) 在相机坐标系中的坐标:".format(pixel_x, pixel_y), point_3d)

        # 在图像上标记像素点
        # cv2.circle(image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)

    # 计算平均旋转矩阵和平均平移向量
    average_rotation_matrix = sum_rotation_matrix / len(ids)
    average_translation_vector = sum_translation_vector / len(ids)

    # 将平均旋转矩阵转换回旋转向量
    average_rvec, _ = cv2.Rodrigues(average_rotation_matrix)

    print()
    print("Average Rotation Vector:", average_rvec.flatten())
    print("Average Translation Vector:", average_translation_vector)

    # 构建平均变换矩阵
    average_transformation_matrix = np.zeros((4, 4), dtype=np.float64)
    average_transformation_matrix[:3, :3] = average_rotation_matrix
    average_transformation_matrix[:3, 3] = average_translation_vector
    average_transformation_matrix[3, 3] = 1.0

    print("Average Transformation Matrix:", average_transformation_matrix)

    # Average_Transformation_Matrix = np.array([[-0.49262632, -0.50407491, -0.0583373, 0.00643042],
    #                                           [-0.50235654, 0.49289276, -0.09278652, 0.00183109],
    #                                           [0.05650777, -0.00889072, -0.99237892, 0.16757868],
    #                                           [0., 0., 0., 1.]])

    Average_Transformation_Matrix = np.array(average_transformation_matrix)

    pixel_position = camera_to_pixel_coordinates(P_object_plane, K)
    print("图像坐标系中的像素位置:", pixel_position)

    point_in_camera = R_plane_to_camera @ point_on_plane + t_plane_to_camera
    print("平面上的点转换为相机坐标系下的坐标为:", point_in_camera)
    print()

    P_object_plane1 = np.array([0, 0, 0])
    Transformation_Matrix = np.array(Average_Transformation_Matrix)
    P_object_plane_homogeneous2 = np.append(P_object_plane1, 1)

    P_object_camera22 = Transformation_Matrix @ P_object_plane_homogeneous2

    P_object_camera2_3D2 = P_object_camera22[:3]
    np.set_printoptions(suppress=True)
    print("P_object_camera2_3D2", P_object_camera2_3D2)

    # pixel_position = transform_point_to_image(P_object_plane, Average_Transformation_Matrix, K)
    # print("平均像素位置:", pixel_position)

    # P_object_plane_homogeneous = np.append(P_object_plane, 1)
    #
    #
    #
    #
    #
    # P_object_camera2 = Average_Transformation_Matrix @ P_object_plane_homogeneous
    #
    # # 获取前三个数（即3D坐标）
    # P_object_camera2_3D = P_object_camera2[:3]
    # # 设置NumPy打印选项以禁用科学计数法
    # np.set_printoptions(suppress=True)
    # # 平面上的点转换为相机坐标系下的坐标为: [0.00643042 0.00183109 0.16757868]  [0.00643104 0.0018314  0.16757868]
    # print("平均:", P_object_camera2_3D)
    # # [-0.00743693 -0.01604594  0.00162425]
    # # 0.01394236 -0.05228611 -0.15844302

    # # 分析计算误差
    # differ = pixel_position - box_center
    # print("differ", differ)

# 显示结果
cv2.imshow('Detected Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

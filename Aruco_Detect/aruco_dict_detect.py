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
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0, 0, 255), 3) # X轴，蓝色
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 3) # Y轴，绿色
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0), 3) # Z轴，红色

    return img



# 设置相机内参和畸变系数（需要根据您的相机校准结果设置）
camera_matrix = np.array([[434.43981934, 0, 322.23464966],
                          [0, 433.24884033, 236.84153748],
                          [0, 0, 1]])

k1 = -0.05277087
k2 = 0.06000207
p1 = 0.00087849
p2 = 0.00136543
k3 = -0.01997724

dist_coeffs = np.array([k1, k2, p1, p2, k3])

# 创建 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# 创建检测参数
parameters = aruco.DetectorParameters()

# 假设标记的实际大小为 100 毫米（即 0.1 米）
marker_size = 0.1

# 读取图像
image = cv2.imread('data/color_image_20250407-172622.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测标记
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None:
    # 估计姿态 返回旋转向量 rvecs 和平移向量 tvecs
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

    for i in range(len(ids)):
        # 绘制坐标轴
        image= drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size)

        cv2.imshow('Axes', image)


        # 输出标记的旋转和平移向量
        print(f"Marker ID: {ids[i][0]}")
        print(f"Rotation Vector: {rvecs[i].flatten()}")
        print(f"Translation Vector: {tvecs[i].flatten()}")

# 显示结果
cv2.imshow('Detected Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

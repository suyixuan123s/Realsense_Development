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

# 准备棋盘格在世界坐标系中的坐标
obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
obj_points *= square_size

# 读取新拍摄的图像
image_path = '/suyixuan/Intel_Realsense_D435_Datasets/color_image_20241025-170147.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 查找棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

if ret:
    # 使用 solvePnP 计算相机到标定板的旋转和平移矩阵
    ret, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)

    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 构造齐次变换矩阵（相机坐标系到标定板坐标系）
    transformation_camera_to_board = np.eye(4)
    transformation_camera_to_board[:3, :3] = rotation_matrix
    transformation_camera_to_board[:3, 3] = tvec[:, 0]

    # 打印变换矩阵
    print("Transformation Matrix from Camera to Board:")
    print(transformation_camera_to_board)
else:
    print("未能找到棋盘格角点，请确保标定板在视野中且清晰可见。")

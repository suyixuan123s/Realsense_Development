import numpy as np
import cv2
import os

def calibrate_camera_and_find_transform(image_path, chessboard_size, square_size, camera_matrix, dist_coeffs):
    """
    计算相机坐标系到标定板坐标系的转换关系，并在图像上显示棋盘格的原点和坐标轴。
    :param image_path: 包含标定板的图像路径
    :param chessboard_size: 棋盘格尺寸 (宽 x 高)
    :param square_size: 棋盘格每个方格的大小（单位：米）
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :return: 相机坐标系到标定板坐标系的齐次变换矩阵
    """
    # 准备世界坐标系中的棋盘格点
    obj_p = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    obj_p *= square_size

    # 读取标定板图像并查找角点
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not ret:
        print("未能找到角点")
        return None

    # 使用 solvePnP 计算相机到标定板的旋转和平移矩阵
    ret, rvec, tvec = cv2.solvePnP(obj_p, corners, camera_matrix, dist_coeffs)

    if not ret:
        print("无法求解相机到标定板的旋转和平移矩阵")
        return None

    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 构造相机到标定板的齐次变换矩阵
    transformation_camera_to_board = np.eye(4)
    transformation_camera_to_board[:3, :3] = rotation_matrix
    transformation_camera_to_board[:3, 3] = tvec[:, 0]

    # 获取棋盘格的原点（第一个角点）
    origin = tuple(corners[0][0].astype(int))

    # 在图像上绘制棋盘格的原点（红色圆点）
    cv2.circle(image, origin, radius=10, color=(0, 0, 255), thickness=-1)

    # 定义坐标轴长度
    axis_length = 0.1  # 10 厘米

    # 计算坐标轴的终点在图像中的投影位置
    axis_points = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
    image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)

    # 将坐标轴绘制到图像上
    image_points = image_points.reshape(-1, 2)
    cv2.line(image, origin, tuple(image_points[0].astype(int)), (0, 0, 255), 5)  # X 轴 - 红色
    cv2.line(image, origin, tuple(image_points[1].astype(int)), (0, 255, 0), 5)  # Y 轴 - 绿色
    cv2.line(image, origin, tuple(image_points[2].astype(int)), (255, 0, 0), 5)  # Z 轴 - 蓝色

    # 显示带有原点和坐标轴的图像
    cv2.imshow("Chessboard with Origin and Axes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return transformation_camera_to_board

# 示例使用
if __name__ == '__main__':
    chessboard_size = (11, 8)  # 棋盘格尺寸
    square_size = 0.03  # 每个方格的大小为0.03米
    image_path = 'E:\ABB-Project\cc-wrs\ABB_Intel_Realsense\Dataset2\chessboard_image_0.jpg'  # 图像路径
    camera_matrix = np.array([[908.05716124, 0., 640.58062138],
                              [0., 907.14785856, 349.07025268],
                              [0., 0., 1.]])
    dist_coeffs = np.array([[0.12338635, -0.09838498, -0.00406485, -0.00240096, -0.49340807]])

    transformation_camera_to_board = calibrate_camera_and_find_transform(
        image_path,
        chessboard_size,
        square_size,
        camera_matrix,
        dist_coeffs
    )

    if transformation_camera_to_board is not None:
        print("Camera to Chessboard Transformation Matrix:")
        print(transformation_camera_to_board)
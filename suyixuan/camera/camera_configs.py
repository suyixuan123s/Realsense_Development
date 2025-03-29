# author: young
import cv2
import numpy as np

left_camera_matrix = np.array([[
    385.360809167600, 0, 326.439935124233], [0, 385.348854118456, 241.596362585282], [0, 0, 1]])
left_distortion = np.array([[0.00746036445182610, -0.00358232086751385, -0.000421311503155155, 0.00113998387662830, 0]])

right_camera_matrix = np.array(
    [[385.778468661424, 0, 326.309739929908], [0, 385.790097867250, 241.465659082747], [0, 0, 1]])
right_distortion = np.array(
    [[0.00899774214072054, -0.00661251061822051, -0.000714220760572046, 0.000771730834556187, 0]])

# 旋转矩阵R和平移向量T

R = np.array([[0.999999809513483, -8.22506698534670e-05, 0.000611725285347005],
              [8.20948311867422e-05, 0.999999964175465, 0.000254773444595701],
              [-0.000611746218718710, -0.000254723176580762, 0.999999780441310]])
T = np.array([-50.0626719811567, 0.0177521779707264, 0.0190872058108044])

size = (1280, 720)  # open windows size


# 返回的重投影矩阵 Q 可以用于从立体图像对中计算深度图
# R1:左摄像机旋转矩阵, P1:左摄像机投影矩阵, Q:重投影矩阵
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)

# 计算重投影矩阵和旋转矩阵 R 是旋转矩阵 P 是投影矩阵 用于进行三维到二维的映射  Q是重投影矩阵 用于计算深度图

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

if __name__ == "__main__":
    print(R1)
    print("------------------------------------")

    print(Q)
    print("---------------")

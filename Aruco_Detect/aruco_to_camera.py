"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import cv2
import numpy as np
import cv2.aruco as aruco


def fitPlane(points):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, vh = np.linalg.svd(points_centered)
    plane_normal = vh[-1, :]
    plane_d = -np.dot(plane_normal, centroid)
    return plane_normal, plane_d


def RodriguesToRotationMatrix(rvec):
    rotation_matrix = cv2.Rodrigues(rvec)[0]
    return rotation_matrix

def plane_to_camera_transform(plane_normal, points_3d):
    point_on_plane = np.mean(points_3d, axis=0)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(z_axis, plane_normal)
    if np.linalg.norm(rotation_vector) < 1e-6:
        R_plane_to_camera = np.eye(3)
    else:
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, plane_normal), -1.0, 1.0))
        rotation_vector = rotation_vector / np.linalg.norm(rotation_vector) * rotation_angle
        R_plane_to_camera = RodriguesToRotationMatrix(rotation_vector)
    t_plane_to_camera = point_on_plane
    return R_plane_to_camera, t_plane_to_camera


camera_matrix = np.array([[434.43981934, 0, 322.23464966],
                          [0, 433.24884033, 236.84153748],
                          [0, 0, 1]])

k1 = -0.05277087
k2 = 0.06000207
p1 = 0.00087849
p2 = 0.00136543
k3 = -0.01997724

dist_coeffs = np.array([k1, k2, p1, p2, k3])
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
marker_size = 0.025
image = cv2.imread(r'../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/color_image_20250409-165409.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

if ids is not None:
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
    points_3d = np.array([tvecs[i].flatten() for i in range(len(ids))])
    plane_normal, plane_d = fitPlane(points_3d)

    R_plane_to_camera, t_plane_to_camera = plane_to_camera_transform(plane_normal, points_3d)

    fx = 434.43981934
    fy = 433.24884033
    cx = 322.23464966
    cy = 236.84153748

    u, v = 280, 230

    depth_scale = 9.999999747378752e-05

    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    depth_map = cv2.imread('../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/depth_image_20250409-165409.png',
                           cv2.IMREAD_UNCHANGED)

    depth_value = depth_map[v, u] * depth_scale

    print("depth_value:", depth_value)

    # depth_value = 0.27088958  # [-0.02626416 - 0.00389024  0.27088958]

    X = x_norm * depth_value
    Y = y_norm * depth_value
    Z = depth_value

    camera_coords = np.array([X, Y, Z])
    print(f"(u, v)相机坐标系下的三维坐标({u}, {v}) = {camera_coords}")

    P_object_camera = R_plane_to_camera @ camera_coords + t_plane_to_camera
    print("相同的点通过平面到相机坐标系的转换关系得到的坐标为：", P_object_camera)

cv2.imshow('Detected Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

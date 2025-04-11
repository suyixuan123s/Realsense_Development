"""
Author: Yixuan Su
Date: 2025/04/09 14:31
File: demo.py
Description:

"""

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

camera_matrix = np.array([[434.43981934, 0, 322.23464966],
                          [0, 433.24884033, 236.84153748],
                          [0, 0, 1]])

dist_coeffs = np.array([-0.05277087, 0.06000207, 0.00087849, 0.00136543, -0.01997724])

fx = 434.43981934
fy = 433.24884033
cx = 322.23464966
cy = 236.84153748

depth_scale = 9.999999747378752e-05

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
marker_size = 0.025


def detect_aruco_and_transform_coordinates(image, camera_matrix, dist_coeffs, aruco_dict, aruco_params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv2.undistort(gray, camera_matrix, dist_coeffs, None, newcameramtx)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=aruco_params)
    if ids is None or len(ids) == 0:
        print("未检测到 ArUco 标记。")
        return None, image
    marker_centers = {}
    for i, marker_id in enumerate(ids.flatten()):
        marker_centers[marker_id] = np.mean(corners[i][0], axis=0)
    sorted_ids = np.sort(ids.flatten())
    if len(sorted_ids) >= 3:
        origin_id = sorted_ids[0]
        x_axis_id = sorted_ids[1]
        y_axis_id = sorted_ids[2]
        origin = marker_centers[origin_id]
        x_axis_point = marker_centers[x_axis_id]
        y_axis_point = marker_centers[y_axis_id]
        x_axis_length = np.linalg.norm(x_axis_point - origin)
        y_axis_length = np.linalg.norm(y_axis_point - origin)
        print(f"X 轴长度（像素距离）: {x_axis_length}")
        print(f"Y 轴长度（像素距离）: {y_axis_length}")
        x_axis_vector = x_axis_point - origin
        y_axis_vector = y_axis_point - origin
        x_axis_unit_vector = x_axis_vector / np.linalg.norm(x_axis_vector)
        y_axis_unit_vector = np.array([-x_axis_unit_vector[1], x_axis_unit_vector[0]])
        rotation_matrix = np.array([x_axis_unit_vector, y_axis_unit_vector]).T
        marker_centers_transformed = {}
        sorted_marker_ids = sorted(marker_centers.keys())
        for marker_id in sorted_marker_ids:
            center = marker_centers[marker_id]
            translated_center = center - origin
            x_coordinate = np.dot(translated_center, x_axis_unit_vector)
            y_coordinate = np.dot(translated_center, y_axis_unit_vector)
            print("x_coordinate, y_coordinate", x_coordinate, y_coordinate)
            marker_centers_transformed[marker_id] = np.array([x_coordinate, y_coordinate])
    else:
        print("需要至少三个 ArUco 标记才能定义坐标系")
        return None, image
    if len(sorted_ids) > 1:
        origin_id = sorted_ids[0]
        origin_center = marker_centers[origin_id]
        for i in range(1, len(sorted_ids)):
            id = sorted_ids[i]
            center = marker_centers[id]
            cv2.line(image, tuple(origin_center.astype(int)), tuple(center.astype(int)), (0, 255, 0), 2)  # 绿色线
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return marker_centers_transformed, image, origin


def detect_nut(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nut_centers = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) >= 6:
            area = cv2.contourArea(contour)
            if area > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    nut_centers.append((cX, cY))
    return nut_centers


def process_image_NewXY(image, camera_matrix, dist_coeffs, aruco_dict, aruco_params):
    marker_centers_transformed, image, origin = detect_aruco_and_transform_coordinates(image,
                                                                                       camera_matrix,
                                                                                       dist_coeffs,
                                                                                       aruco_dict, aruco_params)

    nut_positions = []
    if marker_centers_transformed is not None:
        nut_centers = detect_nut(image)
        if nut_centers:
            x_min = min(marker_centers_transformed[marker_id][0] for marker_id in marker_centers_transformed)
            x_max = max(marker_centers_transformed[marker_id][0] for marker_id in marker_centers_transformed)
            y_min = min(marker_centers_transformed[marker_id][1] for marker_id in marker_centers_transformed)
            y_max = max(marker_centers_transformed[marker_id][1] for marker_id in marker_centers_transformed)
            for nut_center in nut_centers:
                nut_center = np.array(nut_center, dtype=np.float32)
                translated_center = nut_center - origin
                nut_position_new_coordinate = translated_center
                if x_min <= nut_position_new_coordinate[0] <= x_max and \
                        y_min <= nut_position_new_coordinate[1] <= y_max:
                    nut_positions.append(nut_position_new_coordinate)
                    cv2.circle(image, tuple(nut_center.astype(int)), 5, (0, 255, 0), -1)
                    cv2.putText(image, "Nut", tuple((nut_center + np.array([10, -10])).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    text_position = tuple((nut_center + np.array([10, 20])).astype(int))
                    cv2.putText(image, f"({nut_position_new_coordinate[0]:.2f}, {nut_position_new_coordinate[1]:.2f})",
                                text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, nut_positions, marker_centers_transformed


if __name__ == "__main__":
    camera_matrix = np.array([[434.43981934, 0.0, 318.67144775],
                              [0.0, 434.35751343, 241.73374939],
                              [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([[-0.04251478, 0.12392275, 0.00148438, 0.00089415, -0.23152345]])
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    aruco_params = aruco.DetectorParameters()

    image = cv2.imread(r"../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/color_image_20250411-113817.jpg")
    if image is None:
        print("无法读取图像，请检查路径是否正确。")
    else:
        processed_image, nut_positions, marker_centers_transformed = process_image_NewXY(image, camera_matrix,
                                                                                         dist_coeffs,
                                                                                         aruco_dict, aruco_params)

        if marker_centers_transformed:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image with Connected Centers and Nut Detections")

            plt.subplot(1, 2, 2)
            for marker_id, center in marker_centers_transformed.items():
                plt.plot(center[0], center[1], 'ro')
                plt.text(center[0], center[1], str(marker_id))

            for nut_position in nut_positions:
                plt.plot(nut_position[0], nut_position[1], 'go')
                plt.text(nut_position[0], nut_position[1], "Nut")

            plt.xlabel("X (Transformed)")
            plt.ylabel("Y (Transformed)")
            plt.title("Transformed Marker Centers and Nut Detections")
            plt.grid(True)
            plt.axis('equal')

            plt.tight_layout()

            plt.savefig('original_image_with_connected_centers111.png')  # 保存原始图像子图

            plt.show()
        else:
            print("未检测到足够的 ArUco 标记，无法显示结果。")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if nut_positions:
            print("检测到的螺母位置（新坐标系）:")
            for position in nut_positions:
                print(f"({position[0]:.2f}, {position[1]:.2f})")
        else:
            print("未检测到螺母。")

# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import json

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流


# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)  # 配置depth流
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)  # 配置color流


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流


profile = pipeline.start(config)  # 流程开始4
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧

    depth_frames = frames.get_depth_frame()
    color_frames = frames.get_depth_frame()

    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################

    depth_frames_intrin = depth_frames.profile.as_video_stream_profile().intrinsics
    color_frames_intrin = color_frames.profile.as_video_stream_profile().intrinsics

    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）

    camera_parameters1 = {'fx': depth_frames_intrin.fx, 'fy': depth_frames_intrin.fy,
                          'ppx': depth_frames_intrin.ppx, 'ppy': depth_frames_intrin.ppy,
                          'height': depth_frames_intrin.height, 'width': depth_frames_intrin.width,
                          'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                          }

    camera_parameters2 = {'fx': color_frames_intrin.fx, 'fy': color_frames_intrin.fy,
                          'ppx': color_frames_intrin.ppx, 'ppy': color_frames_intrin.ppy,
                          'height': color_frames_intrin.height, 'width': color_frames_intrin.width,
                          'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                          }

    camera_parameters3 = {'fx': intr.fx, 'fy': intr.fy,
                          'ppx': intr.ppx, 'ppy': intr.ppy,
                          'height': intr.height, 'width': intr.width,
                          'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                          }

    camera_parameters4 = {'fx': depth_intrin.fx, 'fy': depth_intrin.fy,
                          'ppx': depth_intrin.ppx, 'ppy': depth_intrin.ppy,
                          'height': depth_intrin.height, 'width': depth_intrin.width,
                          'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                          }

    # 保存内参到本地
    with open('Dataset_640_480/intr7insics1.json', 'w') as fp:
        json.dump(camera_parameters1, fp)
    #######################################################

    # 保存内参到本地
    with open('Dataset_640_480/intr7insics2.json', 'w') as fp:
        json.dump(camera_parameters2, fp)
    #######################################################

    # 保存内参到本地
    with open('Dataset_640_480/intr7insics3.json', 'w') as fp:
        json.dump(camera_parameters3, fp)
    #######################################################

    # 保存内参到本地
    with open('Dataset_640_480/intr7insics4.json', 'w') as fp:
        json.dump(camera_parameters4, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


if __name__ == "__main__":
    while 1:
        intr, depth_intrin, rgb, depth, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
        # 定义需要得到真实三维信息的像素点（x, y)，本例程以中心点为例
        print("============")
        print(aligned_depth_frame)
        x = 320
        y = 240
        dis = aligned_depth_frame.get_distance(x, y)  # （x, y)点的真实深度值
        print("dis: ", dis)
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y],
                                                            dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
        print(camera_coordinate)

        cv2.imshow('RGB image', rgb)  # 显示彩色图像

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
    cv2.destroyAllWindows()

import os
import datetime
import pyrealsense2 as rs


def get_depth_camera_info(profile):
    """
    获取深度相机的内参信息和深度值比例因子

    参数：
        pipeline: pipeline对象，已经初始化的深度相机

    返回值：
        depth_scale: 深度值的比例因子
        intrinsics: 深度图像的内参信息
    """
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    print("Depth intrinsics:")
    print(f"Width: {depth_intrinsics.width}")
    print(f"Height: {depth_intrinsics.height}")
    print(f"PPX (principal point x): {depth_intrinsics.ppx}")
    print(f"PPY (principal point y): {depth_intrinsics.ppy}")
    print(f"FX (focal length x): {depth_intrinsics.fx}")
    print(f"FY (focal length y): {depth_intrinsics.fy}")
    print(f"Distortion model: {depth_intrinsics.model}")
    print(f"Distortion coefficients: {depth_intrinsics.coeffs}")


if __name__ == "__main__":
    # 初始化RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始捕获
    profile = pipeline.start(config)

    # 获取并打印深度相机信息
    get_depth_camera_info(profile)

    # 停止捕获
    pipeline.stop()

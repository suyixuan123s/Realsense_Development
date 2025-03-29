import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 创建管道对象
pipeline = rs.pipeline()
config = rs.config()

# 配置相机流（启用彩色流）
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)

try:
    # 等待相机准备好并获取一帧图像
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()  # 获取彩色帧

    if not color_frame:
        print("未获取到图像帧")
    else:
        # 将帧转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 显示图像
        cv2.imshow('Captured Image', color_image)

        # 等待按键
        key = cv2.waitKey(0)  # 等待直到按键按下
        if key == 13:  # 如果按下Enter键（ASCII码为13）
            # 保存图像到指定路径
            output_folder = r"E:\ABB\AI\mecheye_python_samples\data"  # 替换为你的保存路径
            os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）
            image_path = os.path.join(output_folder, "captured_image.png")

            cv2.imwrite(image_path, color_image)  # 保存图像
            print(f"图像已保存到: {image_path}")

finally:
    # 停止流并释放资源
    pipeline.stop()
    cv2.destroyAllWindows()

import cv2
import os
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect
import time

from pyrealsense2 import timestamp_domain

from Intrinsic_calibration import image_counter


class Capture2DImage(object):
    def __init__(self):
        self.camera = Camera()
        self.image_counter = 1  # 初始化图像编号计数器
        self.save_directory = r"G:\mecheye_python_samples\suyixuan\Image_datasets"  # 图像保存路径

        # 如果保存路径不存在，创建路径
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"Directory {self.save_directory} created.")

    def capture_2d_image(self):
        frame_2d = Frame2D()  # 创建一个 Frame2D 对象，用于存储采集到的2D图像帧
        show_error(self.camera.capture_2d(frame_2d))  # 使用相机捕获2D图像并将结果存储在 frame_2d 中

        # 判断图像的颜色类型，选择不同的处理方式
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image2d = frame_2d.get_gray_scale_image()  # 如果是单色图像，获取灰度图
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image2d = frame_2d.get_color_image()  # 如果是彩色图像，获取彩色图

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 设置图像保存路径（包括目录和文件名）
        file_name = os.path.join(self.save_directory, f"2DImage_{timestamp}.png")

        # 使用 OpenCV 保存图像
        cv2.imwrite(file_name, image2d.data())
        print(f"Captured and saved the 2D image: {file_name}")

        # 更新图像编号计数器
        self.image_counter += 1


        return image2d.data()  # 返回捕获的图像数据

    def main(self):
        if find_and_connect(self.camera):  # 查找并连接相机
            print("Press 'Enter' to capture a new 2D image. Press 'Esc' to exit.")
            try:
                while True:  # 持续循环，等待用户按下 Enter 键
                    # 捕获图像并实时显示
                    frame_2d = Frame2D()
                    show_error(self.camera.capture_2d(frame_2d))  # 获取相机图像

                    # 判断图像类型
                    if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
                        image2d = frame_2d.get_gray_scale_image()  # 获取灰度图像
                    elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
                        image2d = frame_2d.get_color_image()  # 获取彩色图像

                    # 实时显示相机画面
                    cv2.imshow("Live Camera Feed", image2d.data())

                    # 等待用户按下 Enter 键
                    key = cv2.waitKey(1)  # 等待1ms，检查是否按下按键
                    if key == 27:  # 按下 'Esc' 键退出程序
                        print("Exiting...")
                        break
                    elif key == 13:  # 按下 'Enter' 键捕获图像
                        self.capture_2d_image()  # 捕获并保存图像

            except KeyboardInterrupt:
                print("\nExiting...")
            finally:
                self.camera.disconnect()  # 断开与相机的连接
                cv2.destroyAllWindows()  # 关闭显示窗口
                print("Disconnected from the camera successfully.")


if __name__ == '__main__':
    a = Capture2DImage()
    a.main()

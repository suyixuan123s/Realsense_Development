"""
Author: Yixuan Su
Date: 2024/11/26 21:24
File: crop_image_to_fixed_size.py
Description: 
"""
import cv2

def crop_image_to_fixed_size(image_path, target_width, target_height):
    """
    将图像裁剪到一定的固定大小。
    :param image_path: 输入图像的路径。
    :param target_width: 裁剪后的图像宽度。
    :param target_height: 裁剪后的图像高度。
    :return: 裁剪后的图像，如果输入图像太小，则返回 None。
    """

    # 加载图像
    image = cv2.imread(image_path)

    # 检查图像是否加载成功
    if image is None:
        print("无法加载图像，请检查路径是否正确。")
        return None

    # 获取图像的宽度和高度
    original_height, original_width = image.shape[:2]
    print(original_height, original_width)

    # 检查图像是否足够大
    if original_width < target_width or original_height < target_height:
        print("输入图像太小，无法裁剪到目标大小。")
        return None

    # 计算裁剪起始位置（居中裁剪）
    start_x = (original_width - target_width) // 2
    start_y = (original_height - target_height) // 2

    # 裁剪图像
    cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]

    return cropped_image

# 示例使用
if __name__ == "__main__":
    cropped = crop_image_to_fixed_size("66f6e2a5a37f1ae196dc6ec6df17ff5.jpg", target_width=1320, target_height=1080)
    if cropped is not None:
        cv2.imshow("Cropped Image", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

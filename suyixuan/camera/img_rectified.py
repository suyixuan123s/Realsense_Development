# author: young
import cv2
import camera_configs

img = cv2.imread(r'E:\ABB\AI\mecheye_python_samples\data\combined_44.png')

img_left = img[0:720, 0:1280]
img_right = img[0:720, 1280:2560]

img_left_rectified = cv2.remap(img_left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
img_right_rectified = cv2.remap(img_right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
concat = cv2.hconcat([img_left_rectified, img_right_rectified])

# 获取拼接后图像的大小
concat_size = concat.shape  # 返回形状 (高度, 宽度, 通道数)

print(f"拼接后图像的大小: {concat_size[0]} 像素高, {concat_size[1]} 像素宽")

# 获取拼接后图像的大小
img_size = img.shape  # 返回形状 (高度, 宽度, 通道数)

print(f"拼接后图像的大小: {img_size[0]} 像素高, {img_size[1]} 像素宽")

i = 0
while (i < 720):
    cv2.line(img, (0, i), (2559, i), (0, 255, 0))
    cv2.line(concat, (0, i), (2559, i), (0, 255, 0))
    i = i + 36

cv2.imshow('original', img)
cv2.imshow('rectified', concat)

# 等待用户的按键

cv2.waitKey(0)

# 关闭所有的窗口
cv2.destroyAllWindows()

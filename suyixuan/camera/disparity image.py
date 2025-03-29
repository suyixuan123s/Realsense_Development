# author: young
import cv2
import numpy as np
import camera_configs

# 读取图像
img = cv2.imread('E:\\ABB\\AI\\mecheye_python_samples\\data\\combined_44.png')

# 分割图像为左右部分
img_left = img[0:720, 0:1280]
img_right = img[0:720, 1280:2560]

# 显示左侧原始图像
cv2.imshow("Original Left Image", img_left)

# 图像校正
img_left_rectified = cv2.remap(img_left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
img_right_rectified = cv2.remap(img_right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

# 转换为灰度图像
imgL = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)

# 创建窗口和滑块
cv2.namedWindow('SGBM')
cv2.createTrackbar('numDisparities', 'SGBM', 2, 10, lambda x: None)
cv2.createTrackbar('blockSize', 'SGBM', 5, 255, lambda x: None)

app = 0

while True:
    # 获取滑块位置
    num = cv2.getTrackbarPos('numDisparities', 'SGBM')
    blockSize = cv2.getTrackbarPos('blockSize', 'SGBM')

    # 确保块大小为奇数，且不小于5
    blockSize = max(5, blockSize + (blockSize % 2 == 0))  # 如果是偶数加1

    # 创建StereoSGBM对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * num,  # 确保为16的倍数
        blockSize=blockSize
    )

    # 计算视差图
    dis = stereo.compute(imgL, imgR)
    print(f"dis: {dis}")


    # 打印视差图信息（只在第一次计算时）
    if app == 0:
        print(f"视差图维度: {dis.ndim}")
        print(f"视差图类型: {type(dis)}")
        max_index = np.unravel_index(np.argmax(dis, axis=None), dis.shape)
        app = 1

    # 归一化视差图以便于显示
    dis_normalized = cv2.normalize(dis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 应用颜色映射
    dis_colored = cv2.applyColorMap(dis_normalized, cv2.COLORMAP_JET)  # 使用常量
    cv2.imshow('SGBM', dis_colored)

    # 按1000毫秒显示每帧
    if cv2.waitKey(1000) & 0xFF == 27:  # 按Esc退出
        break

# 清理所有窗口
cv2.destroyAllWindows()

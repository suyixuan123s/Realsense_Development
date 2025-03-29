"""
Author: Yixuan Su
Date: 2024/11/18 13:56
File: Visualization_point_cloud.py

"""

import cv2

# 读取 tiff 文件
img = cv2.imread(r'G:\mecheye_python_samples\area_scan_3d_camera\advanced\RenderedDepthMap.tiff')

# 显示图像
cv2.imshow('Rendered Depth Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# from PIL import Image
#
# # 打开并显示 tiff 文件
# img = Image.open('RenderedDepthMap.tiff')
# img.show()

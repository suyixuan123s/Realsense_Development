from suyixuan.camera.capture_point_cloud import aligned_depth_frame
from suyixuan.camera_ectrinsic.intrinsic import color_intrinsics

# 获取相机的内参
intrinsics = color_intrinsics  # 假设已经获取到了彩色相机的内参
ppx, ppy = intrinsics.ppx, intrinsics.ppy  # 主点坐标
fx, fy = intrinsics.fx, intrinsics.fy      # 焦距

# 假设在彩色图像中检测到的物体中心坐标为(x, y)
x, y = 320, 240  # 例子中的中心点

# 在对齐后的深度图中获取物体中心处的深度值
depth_value = aligned_depth_frame.get_distance(x, y)  # 获取该点的深度值，单位为米

# 计算三维坐标
X = (x - ppx) * depth_value / fx
Y = (y - ppy) * depth_value / fy
Z = depth_value

# 输出物体的三维坐标
print(f"物体的三维坐标为: X={X}, Y={Y}, Z={Z}")

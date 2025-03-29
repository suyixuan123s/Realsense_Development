import open3d

# 定义点云文件路径
point_cloud_path = r'E:\ABB\AI\mecheye_python_samples\area_scan_3d_camera\point_cloud\point_cloud_20241013_113512.ply'

# 读取点云文件
point_cloud = open3d.io.read_point_cloud(point_cloud_path)


# 打印点云信息
print(f"Number of points: {len(point_cloud.points)}")


if point_cloud.has_normals():
    print("Point cloud has normals.")
else:
    print("Point cloud does not have normals.")


if point_cloud.has_colors():
    print("Point cloud has colors.")
else:
    print("Point cloud does not have colors.")


# 可视化点云
open3d.visualization.draw_geometries([point_cloud])






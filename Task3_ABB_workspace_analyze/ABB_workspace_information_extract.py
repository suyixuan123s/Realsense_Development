"""
Author: Yixuan Su
Date: 2025/02/20 14:53
File: ABB_workspace_information_extract.py
Description:

"""

import json

# 读取JSON文件
file_path = 'workspace.json'  # 根据实际路径修改
with open(file_path, 'r') as file:
    data = json.load(file)

# 提取Rack和Tube信息
workspace = data.get("Workspace", {}).get("Table", {}).get("Racks", [])

for rack in workspace:
    print(f"Rack ID: {rack.get('ID')}")
    print(f"Color: {rack.get('Color')}")
    print(f"Is Initialized: {rack.get('isInitialized')}")
    print(f"Type: {rack.get('Type')}")
    print(f"Content: {rack.get('Content')}")
    print(f"Pose: {rack.get('Pose')}")
    print(f"CAD Model Address: {rack.get('CADModelAddress')}")
    print(f"State: {rack.get('State')}")
    print(f"Related Object ID: {rack.get('RelatedObjectID')}")
    print(f"Relative Position: {rack.get('RelativePosition')}")
    print(f"Occupancy of Holes: {rack.get('OccupancyOfHoles')}")
    print(f"Pose of Holes: {rack.get('PoseOfHoles')}")

    # 提取每个Rack中的Tubes信息
    tubes = rack.get('Tubes', [])
    for tube in tubes:
        print(f"  Tube ID: {tube.get('ID')}")
        print(f"  Color: {tube.get('Color')}")
        print(f"  Is Initialized: {tube.get('isInitialized')}")
        print(f"  Type: {tube.get('Type')}")
        print(f"  Content: {tube.get('Content')}")
        print(f"  Pose: {tube.get('Pose')}")
        print(f"  CAD Model Address: {tube.get('CADModelAddress')}")
        print(f"  State: {tube.get('State')}")
        print(f"  Related Object ID: {tube.get('RelatedObjectID')}")
        print(f"  Relative Position: {tube.get('RelativePosition')}")
        print()  # 打印空行以区分不同Tube的信息

    print("-" * 50)  # 用于区分不同Rack的信息

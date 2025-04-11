"""
Author: Yixuan Su
Date: 2025/04/11 14:31
File: demo.py
Description:

"""

import numpy as np

points_3d = np.array([[0.00495591, 0.05540022, 0.37385389],
                     [0.17049057, 0.05350205, 0.38673178],
                     [0.00280393, -0.10366391, 0.40791784],
                     [0.16417337, -0.10958427, 0.40655665]])

average_point = np.mean(points_3d, axis=0)

print(average_point)

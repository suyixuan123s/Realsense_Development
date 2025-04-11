import cv2
import cv2 as cv
import numpy as np
from pyapriltags import Detector
from scipy.spatial.transform import Rotation

img_path = '../Aruco_Detect/data/00000.jpg'
tag_id = 2
tag_size = 0.0755

fx, fy, cx, cy = 1828.541260, 1828.113647, 1928.192749, 1105.186768

intr_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

k1 = 0.428687
k2 = -2.811359
k3 = 1.742696
k4 = 0.310000
k5 = -2.625735
k6 = 1.659176
p1 = 0.000465
p2 = -0.000141

distortion = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

img = cv.imread(img_path)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

at_detector = Detector(searchpath=['apriltags'],
                       families='tag16h5',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

detections = list(filter(lambda detection: detection.tag_id == tag_id, at_detector.detect(gray_img)))

if not detections:
    print('No detection found')
    exit(-1)

detection = detections[0]
print("detection:", detection)
print()

corners = detection.corners

for i in range(4):
    cv.circle(img, np.int16(corners[i]), 1, (0, 0, 255), 9)

cv.imshow('img', img[::2, ::2, ...])
cv.waitKey()

obj_pts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]) * tag_size

_, rvec, tvec = cv.solvePnP(obj_pts, corners, intr_mat, distortion)
rot = Rotation.from_rotvec(rvec.flatten())
# rot, _ = cv2.Rodrigues(rvec)


extr_mat = np.eye(4, dtype=float)
extr_mat[:3, :3] = rot.as_matrix()
# extr_mat[:3, :3] = rot
extr_mat[:3, 3] = tvec.flatten()

print("extr_mat", extr_mat)
print()

# demo1： [1.36841495    3.17775439 -125.43197112]
#
# demo2 [-0.0386275  -0.6351637   0.57172403]


inv_extr_mat = np.linalg.inv(extr_mat)
print("demo1：", Rotation.from_matrix(inv_extr_mat[:3, :3]).as_euler('zyx', degrees=True))
print()
print("demo2", inv_extr_mat[:3, 3])
print()

edges = cv.Canny(gray_img, 100, 200)
cv.imwrite('edges.jpg', edges)

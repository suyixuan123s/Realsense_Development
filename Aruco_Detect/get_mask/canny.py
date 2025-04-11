import cv2

img = cv2.imread('../../Task1_Intel_Realsense_D435/Data_Intel_Realsense_D405/color_image_20250410-172926.jpg')

img_Gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_Guass = cv2.GaussianBlur(img_Gray, (7, 7), 10)
img_Conny = cv2.Canny(img_Gray, 640, 480)

cv2.imshow('Aruco', img_Conny)
cv2.imshow("Original image", img)
cv2.waitKey()
cv2.destroyAllWindows()

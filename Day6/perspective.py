
import cv2
import numpy as np

image = cv2.imread(r"C:\Users\shirs\Downloads\banana-8719086_1280.jpg")
rows, cols = image.shape[:2]
# Define points for perspective transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [300, 200]])
M = cv2.getPerspectiveTransform(pts1, pts2)

perspective = cv2.warpPerspective(image, M, (cols, rows))

cv2.imshow('Perspective Transformed Image', perspective)
cv2.waitKey(0)
cv2.destroyAllWindows()

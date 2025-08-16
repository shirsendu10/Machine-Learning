import cv2
import numpy as np

image = cv2.imread(r"C:\Users\shirs\Downloads\banana-8719086_1280.jpg")
rows, cols = image.shape[:2]

# Shear matrix
M = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
sheared = cv2.warpAffine(image, M, (cols + 100, rows + 100))

cv2.imshow('Sheared Image', sheared)
cv2.waitKey(0)
cv2.destroyAllWindows()

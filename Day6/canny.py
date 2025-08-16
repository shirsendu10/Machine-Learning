import cv2
import numpy as np

image = cv2.imread(r"C:\Users\shirs\Downloads\banana-8719086_1280.jpg")


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)  # Lower and upper threshold

cv2.imshow('Edge Detected Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

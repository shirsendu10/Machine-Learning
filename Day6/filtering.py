
import cv2
import  numpy as np

image = cv2.imread(r"C:\Users\shirs\Downloads\banana-8719086_1280.jpg")
# Blurring
blurred = cv2.GaussianBlur(image, (7, 7), 0)

# Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)

cv2.imshow('Blurred Image', blurred)
cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()

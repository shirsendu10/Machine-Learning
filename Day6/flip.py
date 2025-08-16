
import cv2
import numpy as np

image = cv2.imread(r"C:\Users\shirs\Downloads\banana-8719086_1280.jpg")

# Horizontal Flip
flipped_h = cv2.flip(image, 1)

# Vertical Flip
flipped_v = cv2.flip(image, 0)

# Both Axes Flip
flipped_both = cv2.flip(image, -1)

cv2.imshow('Flipped Horizontally', flipped_h)
cv2.imshow('Flipped Vertically', flipped_v)
cv2.imshow('Flipped Both', flipped_both)
cv2.waitKey(0)
cv2.destroyAllWindows()

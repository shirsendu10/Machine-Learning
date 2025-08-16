import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread(r'C:\Users\shirs\Downloads\banana-8719086_1280.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print("Error: Image not loaded.")
    exit()

# Apply the Sobel operator in the x direction (horizontal gradient)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Apply the Sobel operator in the y direction (vertical gradient)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude of gradients
magnitude = cv2.magnitude(sobel_x, sobel_y)

# Convert the magnitude to a format that can be displayed (uint8)
magnitude = np.uint8(np.absolute(magnitude))

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Sobel X Gradient", sobel_x)
cv2.imshow("Sobel Y Gradient", sobel_y)
cv2.imshow("Magnitude of Gradients", magnitude)

# Wait until a key is pressed and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

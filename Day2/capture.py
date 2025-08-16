import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a frame
ret, frame = cap.read()

# Check if the frame was captured successfully
if ret:
    # Save the frame as an image
    cv2.imwrite("captured_image.jpg", frame)

    # Display the captured image (optional)
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()
else:
    print("Error: Could not capture frame.")

# Release the camera
cap.release()
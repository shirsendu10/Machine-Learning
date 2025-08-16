import cv2
import numpy as np

#Initiliaze global variables
ref_point = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping,clone

    #If the left mouse button is pressed, record the starting (x,y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x,y)]
        cropping = True

    #If the left mouse button is released, record the ending (x,y) coordinates
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x,y))
        cropping = False

        #Draw a rectangle around the selected region
        cv2.rectangle(clone, ref_point[0], ref_point[1], (0,255,0), 2)
        cv2.imshow("image", clone)

#Load the image
image = cv2.imread(r"C:\Users\shirs\Downloads\banana-8719086_1280.jpg")

if image is None:
    print("Error:Image not loaded.Check the file path.")
    exit()

clone = image.copy()

#Create a window and set the mouse callback function
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

print("Click and drag to select a region.")
print("Press 'c' to crop, 's' to save the cropped image , 'r' to reset, or 'q' to quit")

cropped_image= None #Intialize cropped image

while True:
    #Display the image 
    cv2.imshow("image", clone)
    key= cv2.waitKey(1) & 0xFF

    #If 'r' is pressed, reset the cropping region
    if key == ord("r"):
        clone = image.copy()
        ref_point = []
        cropped_image=None
        print("Selection reset.Draw a new region.")
    #If 'c' is pressed, crop the selected region
    elif key == ord("c"):
        if len(ref_point) == 2:
            #Crop the region of interest
            x_start, y_start = ref_point[0]
            x_end, y_end = ref_point[1]

            #Ensure coordinates are in correct order
            x_start, x_end = min(x_start, x_end), max(x_start, x_end)
            y_start, y_end = min(y_start, y_end), max(y_start, y_end)

            cropped_image = image[y_start:y_end, x_start:x_end]
            cv2.imshow("Cropped Image", cropped_image)
            print(f"Cropped region: x={x_start} , y={y_start}, width={x_end-x_start}, height={y_end-y_start}")
        else:
            print("please select a region first!")

    #If 's' is pressed, save the cropped image 
    elif key == ord("s"):
        if cropped_image is not None:
            filename = "cropped_image.jpg" #Default filename      
            cv2.imwrite(filename, cropped_image)
            print(f"Cropped image saved as {filename}")
        else:
            print("No cropped image to save. Please crop an image first!")

    #If 'q' is pressed, exit the loop
    elif key == ord("q"):
           print("Exiting...")
           break

cv2.destroyAllWindows()
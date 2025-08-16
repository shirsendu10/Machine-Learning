import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import matplotlib.pyplot as plt
import os
#image path
image_path = r"C:\Users\shirs\OneDrive\Pictures\Camera Roll\WIN_20250107_16_45_59_Pro.jpg"

#1. Check if the image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: Image file not found at path: {image_path} ")

#2. Load the RGB image 
rgb_image = cv2.imread(image_path)
if rgb_image is None:
    raise ValueError(f"Error: Unable to load image from path: {image_path}")

rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) #convert BGR to RGB

#3. RGB to Grayscale
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray_image.jpg', gray_image)
#cv2.imshow('Grayscale Image', gray_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#4. Gray to Binary
_,binary_image= cv2.threshold(gray_image, 125, 255, cv2.THRESH_BINARY) #black means 0 white stand by 1
cv2.imwrite('binary_image.jpg', binary_image)
 
#5. RGB image to Pixel value
height, width, channels =rgb_image.shape
print(f"Image Dimension: {width}X{height}, Channels: {channels}")
x,y = 50,50 # Example pixel coordinates
if x < width and y < height:
    pixel_value = rgb_image[y, x]
    print(f"Pixel value at ({x}, {y}): {pixel_value}")
else:
    print(f"Coordinates ({x}, {y}) are out of image bounds.")

#6. Image Histogram
plt.figure(figsize=(10, 5))
color =('r' ,'g', 'b')
for i,col in enumerate(color):
    hist= cv2.calcHist([rgb_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for RGB Image')
plt.xlabel('pixel value')
plt.ylabel('Frequence')
plt.show() 

#7. Pixel Manipulation
for i in range(min(50,height)):
    for j in range(min(50, width)):
        rgb_image[i,j] = [255, 0 , 0] # Red color
cv2.imwrite('manipulated_image.jpg',cv2.cvtColor(rgb_image,cv2.COLOR_RGB2BGR))

# 8. Metadata Extraction
image =Image.open(image_path)       
exif_data= image._getexif()
if exif_data is not None:
    print("\n Image Metadata")
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        print(f"(tag): (value)")
else:
    print("\n No Metadata Found.")        
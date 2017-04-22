import cv2
import numpy as np

img = cv2.imread("datasets/TrainIJCNN2013/00002.ppm");
#template = cv2.imread("datasets/GTSRB/Final_Training/Images/00001/00000_00012.ppm")
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,80,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,80,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# define range of blue color in HSV
lower_blue = np.array([110, 80, 50])
upper_blue = np.array([130, 255, 255])

mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
# set my output img to zero everywhere except my mask
output_img_red = img.copy()
output_img_red[np.where(mask==0)] = 0
output_img_red = cv2.cvtColor(output_img_red,cv2.COLOR_BGR2GRAY)

output_img_blue = img.copy()
output_img_blue[np.where(mask_blue==0)] = 0
output_img_blue = cv2.cvtColor(output_img_blue,cv2.COLOR_BGR2GRAY)

cv2.imshow('img', output_img_red)
cv2.imshow('img2',output_img_blue)
cv2.imshow('img3',template)
cv2.waitKey(0)
cv2.destroyAllWindows()
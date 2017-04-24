import cv2
import numpy as np
from matplotlib import pyplot as plt

def canny_edge(im, sigma=0.33):

    im = cv2.imread("datasets/TestIJCNN2013/00021.ppm")
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    v = np.median(im)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(max(0, (1.0 + sigma) * v))

    # Apply the canny edge detector to the image
    edges = cv2.Canny(gray_image, lower, upper)

    # Check for circles using Hough transform
    blur = cv2.GaussianBlur(gray_image,(3,3),2,2)
    # (image, method, dp, minDist, high threshold for canny, accumulator threshold for circle centers, minRadius, maxRadius)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT,1,100,param1=upper,param2=5,minRadius=0,maxRadius=100)
    print("after houghcircles")
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(gray_image,(i[0],i[1]),i[2],(0,255,0),2)

    plt.subplot(121), plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


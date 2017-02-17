import numpy as np
import cv2
import colorsys
from matplotlib import pyplot as plt

img = cv2.imread('coins.jpg')

b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Blurring image
grayBlur = cv2.medianBlur(gray, 3)

# Binary threshold
ret, thresh = cv2.threshold(grayBlur, 200,255, cv2.THRESH_BINARY_INV)

# Noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=1)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)

# Coloring borders black
img[markers == -1] = [0,0,0]

#Color background white
img[markers == 1] = [255, 255, 255]

# Color nodes
nodes = np.amax(markers)
for  i in range(nodes):
    (r, g, b) = colorsys.hsv_to_rgb(float(i) / nodes, 1.0, 1.0)
    R, G, B = int(255 * r), int(255 * g), int(255 * b)
    color = [R,G,B]
    print(color)
    img[markers == i+2] = list(color)

# Add text with coin count
text = 'Coins: ' + (str)(np.amax(markers-1))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,text,(160,20), font, 0.5,(0,0,0),1,cv2.LINE_AA)


# Plotting
plt.subplot(321), plt.imshow(rgb_img  )
plt.title('Input image'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(thresh, 'gray')
plt.title("Binary threshold"), plt.xticks([]), plt.yticks([])

plt.subplot(323),plt.imshow(sure_bg, 'gray')
plt.title("Sure background"), plt.xticks([]), plt.yticks([])

plt.subplot(324),plt.imshow(sure_fg, 'gray')
plt.title("Sure foreground"), plt.xticks([]), plt.yticks([])

plt.subplot(325),plt.imshow(dist_transform, 'gray')
plt.title("Distance transform"), plt.xticks([]), plt.yticks([])
plt.subplot(326),plt.imshow(img, 'gray')
plt.title("Result from watershed"), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

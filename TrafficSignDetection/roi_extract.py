import cv2
import numpy as np
from App import pyramid


templates_path = "template_images/"
red_triangle_path = "red_templates/triangles/"
red_circle_path = "red_templates/circles/"
blue_path = "blue_templates/"
num_blue_templates = 8
num_red_triangles = 5
num_red_circles = 2
tW = 16
tH = 16
red_circle_templates = []
red_triangle_templates = []
blue_templates = []

for i in range(num_blue_templates):
    blue_templates.append(cv2.imread(templates_path + blue_path + str(i)+".png"))

for j in range(num_red_circles):
    red_circle_templates.append(cv2.imread(templates_path + red_circle_path + str(j)+".png"))

for k in range(num_red_triangles):
    red_triangle_templates.append(cv2.imread(templates_path + red_triangle_path + str(k)+".png"))



img = cv2.imread("datasets/TrainIJCNN2013/00123.ppm");
#template = cv2.imread("datasets/GTSRB/Final_Training/Images/00001/00000_00012.ppm")
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('original',img)
cv2.waitKey(0)
# lower mask (0-10)
lower_red = np.array([0,50,10])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([160,50,10])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# define range of blue color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
# set my output img to zero everywhere except my mask
output_img_red = img.copy()
output_img_red[np.where(mask==0)] = 0
output_img_red = cv2.cvtColor(output_img_red,cv2.COLOR_BGR2GRAY)

output_img_blue = img.copy()
output_img_blue[np.where(mask_blue==0)] = 0
output_img_blue = cv2.cvtColor(output_img_blue,cv2.COLOR_BGR2GRAY)

thresh_circles = 0.5
thresh_triangles = 0.55
thresh_blue = 0.6

found_roi = []

for im in red_circle_templates:

    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    for resized in pyramid(output_img_red, scale = 1.2, minSize=(100, 100)):
        #save the scale
        r = output_img_red.shape[1]/float(resized.shape[1])
        res = cv2.matchTemplate(resized, im, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thresh_circles)
        #cv2.imshow('detected', resized)
        #cv2.waitKey(0)
        for pt in zip(*loc[::-1]):
            #store the found region of interest, scaled up to original image size
            found_roi.append([int(pt[0] * r), int(pt[1] * r), int((pt[0] + tW) * r), int((pt[1] + tH) * r)])
            #cv2.rectangle(resized, pt, (pt[0] + tW, pt[1] + tH), (255, 255, 255), 1)

for im in red_triangle_templates:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for resized in pyramid(output_img_red, scale=1.2, minSize=(100, 100)):
        # save the scale
        r = output_img_red.shape[1] / float(resized.shape[1])
        res = cv2.matchTemplate(resized, im, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thresh_triangles)
        #cv2.imshow('detected', resized)
        #cv2.waitKey(0)
        for pt in zip(*loc[::-1]):
            # store the found region of interest, scaled up to original image size
            found_roi.append([int(pt[0] * r), int(pt[1] * r), int((pt[0] + tW) * r), int((pt[1] + tH) * r)])
            #cv2.rectangle(resized, pt, (pt[0] + tW, pt[1] + tH), (255, 255, 255), 1)

for im in blue_templates:
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for resized in pyramid(output_img_blue, scale=1.2, minSize=(100, 100)):
        # save the scale
        r = output_img_red.shape[1] / float(resized.shape[1])
        res = cv2.matchTemplate(resized, im, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= thresh_blue)
        #cv2.imshow('detected', resized)
        #cv2.waitKey(0)
        for pt in zip(*loc[::-1]):
            # store the found region of interest, scaled up to original image size
            found_roi.append([int(pt[0] * r), int(pt[1] * r), int((pt[0] + tW) * r), int((pt[1] + tH) * r)])
            #cv2.rectangle(resized, pt, (pt[0] + tW, pt[1] + tH), (255, 255, 255), 1)


#cv2.imshow('img', output_img_red)
#cv2.imshow('img2',output_img_blue)
found_roi_copy = found_roi.copy()
for roi in found_roi_copy:
    found_roi.append(roi)

rectList, weights = cv2.groupRectangles(found_roi, 1, 0.04)
print(len(rectList))
for region in rectList:
    print("Found rectangle", region[0], region[1])
    cv2.rectangle(img, (region[0], region[1]),(region[2], region[3]), (255, 0, 255), 2)
cv2.imshow('Original image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
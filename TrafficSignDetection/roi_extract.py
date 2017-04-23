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


def ROI(img):

    #img = cv2.imread("datasets/TrainIJCNN2013/00123.ppm");
    #template = cv2.imread("datasets/GTSRB/Final_Training/Images/00001/00000_00012.ppm")
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('original',img)
    #cv2.waitKey(0)
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
    #print(len(rectList))
    #for region in rectList:
    #    print("Found rectangle", region[0], region[1])
    #    cv2.rectangle(img, (region[0], region[1]),(region[2], region[3]), (255, 0, 255), 2)

    X = []
    for rect in rectList:
        #im = img[np.int(rect[0]):np.int(rect[3]),np.int(rect[1]):np.int(rect[2])]
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        im = img[y1:y2, x1:x2]
        X.append(im)

    X = _extract_feature(X)
    return  rectList, X
    #cv2.imshow('Original image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def test():
    img = cv2.imread("datasets/TrainIJCNN2013/00123.ppm");
    # template = cv2.imread("datasets/GTSRB/Final_Training/Images/00001/00000_00012.ppm")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    # lower mask (0-10)
    lower_red = np.array([0, 50, 10])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([160, 50, 10])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    # set my output img to zero everywhere except my mask
    output_img_red = img.copy()
    output_img_red[np.where(mask == 0)] = 0
    output_img_red = cv2.cvtColor(output_img_red, cv2.COLOR_BGR2GRAY)

    output_img_blue = img.copy()
    output_img_blue[np.where(mask_blue == 0)] = 0
    output_img_blue = cv2.cvtColor(output_img_blue, cv2.COLOR_BGR2GRAY)

    thresh_circles = 0.5
    thresh_triangles = 0.55
    thresh_blue = 0.6

    found_roi = []

    for im in red_circle_templates:

        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for resized in pyramid(output_img_red, scale=1.2, minSize=(100, 100)):
            # save the scale
            r = output_img_red.shape[1] / float(resized.shape[1])
            res = cv2.matchTemplate(resized, im, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= thresh_circles)
            # cv2.imshow('detected', resized)
            # cv2.waitKey(0)
            for pt in zip(*loc[::-1]):
                # store the found region of interest, scaled up to original image size
                found_roi.append([int(pt[0] * r), int(pt[1] * r), int((pt[0] + tW) * r), int((pt[1] + tH) * r)])
                # cv2.rectangle(resized, pt, (pt[0] + tW, pt[1] + tH), (255, 255, 255), 1)

    for im in red_triangle_templates:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for resized in pyramid(output_img_red, scale=1.2, minSize=(100, 100)):
            # save the scale
            r = output_img_red.shape[1] / float(resized.shape[1])
            res = cv2.matchTemplate(resized, im, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= thresh_triangles)
            # cv2.imshow('detected', resized)
            # cv2.waitKey(0)
            for pt in zip(*loc[::-1]):
                # store the found region of interest, scaled up to original image size
                found_roi.append([int(pt[0] * r), int(pt[1] * r), int((pt[0] + tW) * r), int((pt[1] + tH) * r)])
                # cv2.rectangle(resized, pt, (pt[0] + tW, pt[1] + tH), (255, 255, 255), 1)

    for im in blue_templates:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for resized in pyramid(output_img_blue, scale=1.2, minSize=(100, 100)):
            # save the scale
            r = output_img_red.shape[1] / float(resized.shape[1])
            res = cv2.matchTemplate(resized, im, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= thresh_blue)
            # cv2.imshow('detected', resized)
            # cv2.waitKey(0)
            for pt in zip(*loc[::-1]):
                # store the found region of interest, scaled up to original image size
                found_roi.append([int(pt[0] * r), int(pt[1] * r), int((pt[0] + tW) * r), int((pt[1] + tH) * r)])
                # cv2.rectangle(resized, pt, (pt[0] + tW, pt[1] + tH), (255, 255, 255), 1)


    # cv2.imshow('img', output_img_red)
    # cv2.imshow('img2',output_img_blue)
    found_roi_copy = found_roi.copy()
    for roi in found_roi_copy:
        found_roi.append(roi)

    rectList, weights = cv2.groupRectangles(found_roi, 1, 0.04)
    print(len(rectList))
    for region in rectList:
        print("Found rectangle", region[0], region[1])
        cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (255, 0, 255), 2)
    cv2.imshow('Original image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _extract_feature(X, feature='hog'):
    """Performs feature extraction

        :param X:       data (rows=images, cols=pixels)
        :param feature: which feature to extract
                        - None:   no feature is extracted
                        - "gray": grayscale features
                        - "rgb":  RGB features
                        - "hsv":  HSV features
                        - "surf": SURF features
                        - "hog":  HOG features
        :returns:       X (rows=samples, cols=features)
    """

    # transform color space
    if feature == 'gray' or feature == 'surf':
        X = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]
    elif feature == 'hsv':
        X = [cv2.cvtColor(x, cv2.COLOR_BGR2HSV) for x in X]

    # operate on smaller image
    small_size = (32, 32)
    X = [cv2.resize(x, small_size) for x in X]

    # extract features
    if feature == 'surf':
        surf = cv2.SURF(400)
        surf.upright = True
        surf.extended = True
        num_surf_features = 36

        # create dense grid of keypoints
        dense = cv2.FeatureDetector_create("Dense")
        kp = dense.detect(np.zeros(small_size).astype(np.uint8))

        # compute keypoints and descriptors
        kp_des = [surf.compute(x, kp) for x in X]

        # the second element is descriptor: choose first num_surf_features
        # elements
        X = [d[1][:num_surf_features, :] for d in kp_des]
    elif feature == 'hog':
        # histogram of gradients
        block_size = (small_size[0] // 2, small_size[1] // 2)
        block_stride = (small_size[0] // 4, small_size[1] // 4)
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor(small_size, block_size, block_stride,
                                cell_size, num_bins)
        X = [hog.compute(x) for x in X]
    elif feature is not None:
        # normalize all intensities to be between 0 and 1
        X = np.array(X).astype(np.float32) / 255

        # subtract mean
        X = [x - np.mean(x) for x in X]

    X = [x.flatten() for x in X]

    return X
import cv2
import numpy
import utils
from matplotlib import pyplot as plt


def g_hpf(src, kernelSize):
    graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(graySrc, (kernelSize, kernelSize), 0)
    src = graySrc - blurred
    return src


"""
def strokeEdges(src, dst, edgeKsize, blurKsize = 7):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)
"""


def strokeEdges(src, dst, edgeKsize, blurKsize=7):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        cv2.imshow("blurredSrc", blurredSrc)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    cv2.imshow("Laplacian", graySrc)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    cv2.imshow("normalizedInverseAlpha", normalizedInverseAlpha)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


def edgeLaplacian(src, edgeKlSize, blurKsize):
    blurredSrc = cv2.medianBlur(src, blurKsize)
    graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKlSize)
    return graySrc


def edgeLaplacianColor(src, edgeKlSize, blurKsize):
    blurredSrc = cv2.medianBlur(src, blurKsize)
    graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKlSize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, src)


def cannyEdge(src, thr1, thr2):
    return cv2.Canny(src, thr1, thr2)


def contourDetection(src):
    grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayImage, 127, 255, 0)

    # cv2.imshow("thresh", thresh)
    # cv2.imshow("src", src)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("image", image)

    return cv2.drawContours(src, contours, -1, (0, 255, 0), 2)


def contourDetectionPolygon(src):
    img = src
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    colorThresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 800 and area < 20000:
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            hull = cv2.convexHull(c)
            cv2.polylines(colorThresh, [approx], True, (0, 0, 255), 2)
            cv2.polylines(colorThresh, [hull], True, (0, 255, 0), 2)

    return colorThresh


def contourDetectionCircleSquare(src):
    img = src
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    colorThresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 800 and area < 5000:
            # find bounding box coordinates
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(colorThresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # find minimum area
            rect = cv2.minAreaRect(c)
            # calculate coordinates of the minimum area rectangle
            box = cv2.boxPoints(rect)
            # normalize coordinates to integers
            box = numpy.int0(box)
            # draw contours
            cv2.drawContours(colorThresh, [box], -1, (0, 0, 255), 3)

            # calculate center and radius of minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            # cast to integers
            center = (int(x), int(y))
            radius = int(radius)
            # draw the circle
            colorThresh = cv2.circle(colorThresh, center, radius, (0, 255, 0), 2)

            colorThresh = cv2.drawContours(colorThresh, c, -1, (255, 0, 0), 3)

    return colorThresh


def circleDetection(src):
    gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 9)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200, param1=200, param2=30, minRadius=2, maxRadius=200)
    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(src, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(src, (i[0], i[1]), 2, (0, 0, 255), 3)




def autoThresh(src, blockSize, blurKern):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    grayBlur = cv2.medianBlur(gray, blurKern)
    ret, thresh = cv2.threshold(grayBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # th2 = cv2.adaptiveThreshold(grayBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize,2)
    th2 = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, 2)

    th3 = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    # cv2.imshow("TH2", th2)
    # cv2.imshow("TH3", th3)

    return th2


def autoThreshMorph(src, blockSize, blurKern):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    grayBlur = cv2.medianBlur(gray, blurKern)
    ret, thresh = cv2.threshold(grayBlur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # th2 = cv2.adaptiveThreshold(grayBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize,2)
    th2 = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, 2)

    kernel = numpy.ones((3, 3), numpy.uint8)
    closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=2)
    # closing = cv2.erode(th2,kernel, iterations = 2)

    cv2.imshow("Morph", closing)

    th3 = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    # cv2.imshow("TH2", th2)
    # cv2.imshow("TH3", th3)

    return th2


def watershed(src, thresh1, blurKern):
    # Show histogram
    #plt.hist(src.ravel(),256,[0,256]); plt.show()

    img = src
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, thresh = cv2.threshold(gray,thresh1,255,cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = numpy.ones((3,3),numpy.uint8)
    threshBlur = cv2.medianBlur(thresh, blurKern)

    """ """
    grayBlur = cv2.medianBlur(gray, blurKern)
    thresh = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh1, 2)
    threshBlur = cv2.medianBlur(thresh, 1)
    """ """
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("threshBlur", threshBlur)

    ret, markers = cv2.connectedComponents(threshBlur)
    markers = markers + 1

    markers = cv2.watershed(img, markers)

    width, height, channel = img.shape

    img = numpy.zeros((width, height), dtype=numpy.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img[markers == -1] = [255,0,0]
    img[markers == 0] = [255, 255, 255]
    # img[markers == 1] = [255, 255, 255]
    img[markers == 2] = [0, 208, 250]
    img[markers == 3] = [253, 9, 181]
    img[markers == 4] = [253, 244, 4]
    img[markers == 5] = [1, 228, 31]
    img[markers == 6] = [35, 1, 228]
    img[markers == 7] = [254, 67, 101]
    img[markers == 8] = [113, 0, 250]
    img[markers == 9] = [253, 140, 0]
    img[markers == 10] = [4, 253, 88]
    img[markers == 11] = [1, 135, 228]
    img[markers == 12] = [228, 1, 14]
    img[markers == 13] = [247, 0, 250]
    img[markers == 14] = [253, 227, 12]
    img[markers == 15] = [4, 253, 134]
    img[markers == 16] = [1, 57, 228]
    img[markers == 17] = [228, 84, 1]
    img[markers == 18] = [217, 91, 67]
    img[markers == 19] = [192, 41, 66]
    img[markers == 20] = [84, 36, 55]
    img[markers == 21] = [83, 119, 122]
    img[markers == 22] = [207, 240, 158]
    img[markers == 23] = [168, 219, 168]
    img[markers == 24] = [121, 189, 154]
    img[markers == 25] = [59, 134, 134]
    img[markers == 26] = [11, 72, 107]
    img[markers == 27] = [119, 79, 56]
    img[markers == 28] = [224, 142, 121]
    img[markers == 29] = [241, 212, 175]
    img[markers == 30] = [236, 229, 206]
    img[markers == 31] = [197, 224, 220]
    img[markers == 32] = [209, 242, 165]
    img[markers == 33] = [239, 250, 180]
    img[markers == 34] = [255, 196, 140]
    img[markers == 35] = [255, 159, 128]
    img[markers == 36] = [245, 105, 145]
    img[markers == 37] = [232, 221, 203]
    img[markers == 38] = [205, 179, 128]
    img[markers == 39] = [3, 101, 100]
    img[markers == 40] = [3, 54, 73]
    img[markers == 41] = [3, 22, 52]
    img[markers == 42] = [73, 10, 61]
    img[markers == 43] = [189, 21, 80]
    img[markers == 44] = [233, 127, 2]
    img[markers == 45] = [248, 202, 0]
    img[markers == 46] = [138, 155, 15]
    img[markers == 47] = [63, 184, 175]
    img[markers == 48] = [127, 199, 175]
    img[markers == 49] = [218, 216, 167]
    img[markers == 50] = [255, 158, 157]
    img[markers == 51] = [255, 61, 127]
    img[markers == 52] = [52, 56, 56]
    img[markers == 53] = [0, 95, 107]
    img[markers == 54] = [0, 140, 158]
    img[markers == 55] = [0, 180, 204]
    img[markers == 56] = [0, 223, 252]
    img[markers == 57] = [255, 78, 80]
    img[markers == 58] = [252, 145, 58]
    img[markers == 59] = [249, 212, 35]
    img[markers == 60] = [237, 229, 116]
    img[markers == 61] = [225, 245, 196]
    img[markers == 62] = [0, 168, 198]
    img[markers == 63] = [64, 192, 203]
    img[markers == 64] = [249, 242, 231]
    img[markers == 65] = [174, 226, 57]
    img[markers == 66] = [143, 190, 0]
    img[markers == 67] = [53, 19, 48]
    img[markers == 68] = [66, 66, 84]
    img[markers == 69] = [100, 144, 138]
    img[markers == 70] = [232, 202, 164]
    img[markers == 71] = [204, 42, 65]
    img[markers == 72] = [255, 153, 0]
    img[markers == 73] = [66, 66, 66]
    img[markers == 74] = [233, 233, 233]
    img[markers == 75] = [188, 188, 188]
    img[markers == 76] = [207, 190, 39]
    img[markers == 77] = [242, 116, 53]
    img[markers == 78] = [240, 36, 117]
    img[markers == 79] = [59, 45, 56]

    return img


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)

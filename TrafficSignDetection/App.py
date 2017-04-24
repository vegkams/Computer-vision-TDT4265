#!/usr/bin/env python
# -*- coding: utf-8 -*-


import imutils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import os
import roi_extract
import time

import gtsrb
from classifiers import MultiClassSVM

plotComparisson = False

#cam = cv2.VideoCapture(0)


def main():
    #createNegativeTrainingSet()
    MCS = train_mcs()
    #img = cv2.imread("datasets/TrainIJCNN2013/00003.ppm")

    cap = cv2.VideoCapture('Videos/plumbus.mp4')


#    rootpath = "datasets/TrainIJCNN2013"
#    prefix = rootpath + "/"
#    # annotations file
#    gt_file = open(rootpath + "/gt.txt")
#
#    # csv parser for annotations file
#    gt_reader = csv.reader(gt_file, delimiter=';')
#    row = gt_reader.__next__()
#    im = cv2.imread(prefix + row[0])
#    label = np.array(np.int(row[5]))
#    winl = 32
#    winh = 32
#
#    for resized in pyramid(im, scale = 1.5):
#        for (x, y, window) in sliding_window(im, stepSize=8,windowSize=(winl,winh)):
#            if window.shape[0] != winh or window.shape[1] != winl:
#                continue
#            x = gtsrb._extract_feature(window, 'hog')
#            x =  np.squeeze(np.array(x)).astype(np.float32)
#            acc, prec, rec = MCS.evaluate(x, label)
#            print("-accuracy: ", acc)
# Implement non-maxima suppression or something
    imagesBad = load_images_from_folder("datasets/bad_fake/")
    imagesGood = load_images_from_folder("datasets/good_fake/")
#
    #for img in imagesGood:
    #    start_time = time.time()
    #    roi, X = roi_extract.ROI(img)
    #    MCS.evaluateLive(X, roi, img)
    #    print("--- %s seconds ---" % (time.time() - start_time))
    #    if cv2.waitKey(1) == 27:
    #        break  # esc to quit
#
    #for img in imagesBad:
    #    roi, X = roi_extract.ROI(img)
    #    MCS.evaluateLive(X, roi, img)
    #    filename = "{0}.jpg".format(counter)
    #    cv2.imwrite(filename, img)
    #    counter += 1
    #    if cv2.waitKey(1) == 27:
    #        break  # esc to quit

    while (cap.isOpened()):
        start_time = time.time()
        ret, img = cap.read()
        if ret:
            roi, X = roi_extract.ROI(img)
            MCS.evaluateLive(X, roi, img)
            print("--- %s seconds ---" % (time.time() - start_time))
        if cv2.waitKey(1) == 27:
                break  # esc to quit

    while True:
        start_time = time.time()
        img = webcamGrab()
        roi, X = roi_extract.ROI(img)
        MCS.evaluateLive(X, roi, img)
        print("--- %s seconds ---" % (time.time() - start_time))
        #cv2.imshow("Tesla", img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    testData = gtsrb.load_test_data()
    X_test = np.squeeze(np.array(testData[0])).astype(np.float32)
    y_test = np.array(testData[1])
    #MCS.evaluateData(X_test, y_test,X_order=testData[2],picDict=testData[3],signBorders=testData[4],signCounterList=testData[5])


    #X_live = [X_test[1],X_test[2],X_test[3]]
    #X_live_borders = [[983,388,1024,432],[386, 494, 442, 552],[973, 335, 1031, 390]]

    #MCS.evaluateLive(X_live, X_live_borders)
    #roi, X = roi_extract.ROI(img)
    #MCS.evaluateLive(X, roi, img)


    MCS.evaluateLive(X_live, X_live_borders)
    img_test = cv2.imread('datasets/TestIJCNN2013/00107.ppm')
    (winW, winH) = (32, 32)
    for resized in pyramid(img_test, scale=1.5):
        for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            hogResized = gtsrb._extract_feature(resized, 'hog')
            MCS.evaluateLive_sliding(hogResized, [[0, 0, 32, 32]], resized)
            print('evaluated')
            # clone = resized.copy()
            # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            # cv2.imshow("Window", clone)
            # cv2.waitKey(1)

    #show_webcam(True)



def train_mcs():

    #strategies = ['one-vs-one', 'one-vs-all']
    strategies = ['one-vs-one']
    #features = [None, 'gray', 'rgb', 'hsv', 'hog']
    #features = ['hsv', 'hog']
    features = ['hog']
    accuracy = np.zeros((2, len(features)))
    precision = np.zeros((2, len(features)))
    recall = np.zeros((2, len(features)))

    for f in range(len(features)):
        start_time = time.time()
        print("feature", features[f])
        (X_train, y_train), (X_test, y_test) = gtsrb.load_data(feature=features[f], test_split=0.2, seed=42)
        # convert to numpy
        X_train = np.squeeze(np.array(X_train)).astype(np.float32)
        y_train = np.array(y_train)
        X_test = np.squeeze(np.array(X_test)).astype(np.float32)
        y_test = np.array(y_test)

        # find all class labels
        labels = np.unique(np.hstack((y_train, y_test)))

        for s in range(len(strategies)):
            print(" - strategy", strategies[s])
            # set up SVMs
            MCS = MultiClassSVM(len(labels), strategies[s])

            # training phase
            print("    - train")
            MCS.fit(X_train, y_train)

            # test phase
            print("    - test")
            acc, prec, rec = MCS.evaluate(X_test, y_test)
            accuracy[s, f] = acc
            precision[s, f] = np.mean(prec)
            recall[s, f] = np.mean(rec)
            print("       - accuracy: ", acc)
            print("       - mean precision: ", np.mean(prec))
            print("       - mean recall: ", np.mean(rec))
            print("--- %s seconds ---" % (time.time() - start_time))

    # plot results as stacked bar plot

    if plotComparisson:
        f, ax = plt.subplots(2)
        for s in range(len(strategies)):
            x = np.arange(len(features))
            ax[s].bar(x - 0.2, accuracy[s, :], width=0.2, color='b',
                      hatch='/', align='center')
            ax[s].bar(x, precision[s, :], width=0.2, color='r', hatch='\\',
                      align='center')
            ax[s].bar(x + 0.2, recall[s, :], width=0.2, color='g', hatch='x',
                      align='center')
            ax[s].axis([-0.5, len(features) + 0.5, 0, 1.5])
            ax[s].legend(('Accuracy', 'Precision', 'Recall'), loc=2, ncol=3,
                         mode='expand')
            ax[s].set_xticks(np.arange(len(features)))
            ax[s].set_xticklabels(features)
            ax[s].set_title(strategies[s])

        plt.show()

    return MCS
# Create a pyramid of images of decreasing size (iterable method)
def pyramid(image, scale = 1.5, minSize=(30, 30)):
    yield image

    while True:
        w = int(image.shape[1]/scale)
        image = imutils.resize(image,width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

# Slide a window over the image. Returns the image along with it's x and y coordinates
def sliding_window(image, stepSize, windowSize):

    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
	    ret_val, img = cam.read()
	    if mirror:
		    img = cv2.flip(img, 1)
	    cv2.imshow('my webcam', img)
	    if cv2.waitKey(1) == 27:
		    break  # esc to quit
    cv2.destroyAllWindows()

def webcamGrab(mirror=False):
    ret_val, img = cam.read()
    if mirror:
	    img = cv2.flip(img, 1)

    return img


def createNegativeTrainingSet(folderpath = "datasets/TrainIJCNN2013"):
    setpath = "datasets/GTSRB/Final_Training/Images/00044/"
    ofile = open(setpath + 'GT-00044.csv', "w")
    header = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n"
    ofile.write(header)
    counter1 = 0
    counter2 = 0
    Width = 30
    Height = 30
    x = 500
    y = 500

    for filename in os.listdir(folderpath):

        img = cv2.imread(os.path.join(folderpath, filename))
        if img is not None:
            ofile.write('{0:05d}_{1:05d}.ppm;{2};{3};1;1;{4};{5};44\n'.format(counter2,counter1,Width,Height,Width-1,Height-1))

            crop_img = img[y:y + Height, x: x+Width]
            cv2.imwrite(setpath + '/{0:05d}_{1:05d}.ppm'.format(counter2,counter1),crop_img)
            counter1 += 1
            Width += 1
            Height += 1
            if(counter1 == 30):
                Width = 30
                Height = 30
                counter1 = 0
                counter2 += 1
    ofile.close()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

if __name__ == '__main__':
    main()

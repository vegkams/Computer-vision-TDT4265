#!/usr/bin/env python
# -*- coding: utf-8 -*-


import imutils
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv


import gtsrb
from classifiers import MultiClassSVM

plotComparisson = False

def main():

    MCS = train_mcs()

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



    testData = gtsrb.load_test_data()
    X_test = np.squeeze(np.array(testData[0])).astype(np.float32)
    y_test = np.array(testData[1])
    MCS.evaluateData(X_test, y_test,X_order=testData[2],picDict=testData[3],signBorders=testData[4],signCounterList=testData[5])

    MCS.evaluateLive(X_test[5], 0)


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


def train_mcs():
    # strategies = ['one-vs-one', 'one-vs-all']
    strategies = ['one-vs-one']
    # features = [None, 'gray', 'rgb', 'hsv', 'hog']
    features = ['hog']
    accuracy = np.zeros((2, len(features)))
    precision = np.zeros((2, len(features)))
    recall = np.zeros((2, len(features)))

    for f in range(len(features)):
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




if __name__ == '__main__':
    main()

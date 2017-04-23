#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module to load the German Traffic Sign Recognition Benchmark (GTSRB)

    The dataset contains more than 50,000 images of traffic signs belonging
    to more than 40 classes. The dataset can be freely obtained from:
    http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset.
"""

import cv2
import numpy as np

import csv
from matplotlib import cm
from matplotlib import pyplot as plt
import time


def load_test_data(rootpath= "datasets/TrainIJCNN2013",feature='hog'):

    X = []  # data
    labels = []  # corresponding labels
    keyOrder = []    # remeber the order of keys to look up
    picDict = {} # storing classification text
    bordersDict = {} # borders of signs

    signCounterList = []
    signCounter = 1
    prevKey = ''

    rowCounter = 0

    # subdirectory for class
    prefix = rootpath + '/'

    # annotations file
    gt_file = open(rootpath + "/gt.txt")

    # csv parser for annotations file
    gt_reader = csv.reader(gt_file, delimiter=';')

    # loop over all images in current annotations file

    for row in gt_reader:
        # first column is filename
        key = row[0]
        picDict[key] = []

        # Counts signs per image
        if key == prevKey:
            signCounter += 1
            signCounterList.append(signCounter)
        else:
            signCounter = 1
            signCounterList.append(signCounter)
        prevKey = key


        # get all the borders of the signs
        if key not in bordersDict:
            bordersDict[key] = []
            bordersDict[key].append([np.int(row[1]),np.int(row[2]),np.int(row[3]),np.int(row[4]),rowCounter] )
        else:
            bordersDict[key].append([np.int(row[1]), np.int(row[2]), np.int(row[3]), np.int(row[4]),rowCounter])


        keyOrder.append(key)
        im = cv2.imread(prefix + row[0])

        # remove regions surrounding the actual traffic sign
        im = im[np.int(row[2]):np.int(row[4]),
                np.int(row[1]):np.int(row[3]), :]
        X.append(im)
        labels.append(np.int(row[5]))

        rowCounter += 1

    X = _extract_feature(X, feature)
    X_test = X
    Y_test = labels
    return (X_test, Y_test, keyOrder, picDict, bordersDict, signCounterList)



def load_data(rootpath="datasets/GTSRB/Final_Training/Images", feature=None, cut_roi=True,
              test_split=0.2, plot_samples=False, seed=113):
    """Loads the GTSRB dataset

        This function loads the German Traffic Sign Recognition Benchmark
        (GTSRB), performs feature extraction, and partitions the data into
        mutually exclusive training and test sets.

        :param rootpath:     root directory of data files, should contain
                             subdirectories "00000" for all samples of class
                             0, "00004" for all samples of class 4, etc.
        :param feature:      which feature to extract: None, "gray", "rgb",
                             "hsv", surf", or "hog"
        :param cut_roi:      flag whether to remove regions surrounding the
                             actual traffic sign (True) or not (False)
        :param test_split:   fraction of samples to reserve for the test set
        :param plot_samples: flag whether to plot samples (True) or not
                             (False)
        :param seed:         which random seed to use
        :returns:            (X_train, y_train), (X_test, y_test)
    """
    # hardcode available class labels
    #classes = np.array([1, 2])
    #classes = np.array([0, 2, 4, 8, 6, 8, 10, 12, 14, 16, 18, 20, 22, 43])
    #classes = np.array([2,3,4,5])
    classes = np.array(range(0,44))

    # read all training samples and corresponding class labels
    X = []  # data
    labels = []  # corresponding labels
    for c in range(len(classes)):
        # subdirectory for class
        prefix = rootpath + '/' + format(classes[c], '05d') + '/'

        # annotations file
        gt_file = open(prefix + 'GT-' + format(classes[c], '05d') + '.csv')

        # csv parser for annotations file
        gt_reader = csv.reader(gt_file, delimiter=';')
        gt_reader.__next__()  # skip header

        # loop over all images in current annotations file
        for row in gt_reader:
            # first column is filename
            im = cv2.imread(prefix + row[0])

            # remove regions surrounding the actual traffic sign
            if cut_roi:
                im = im[np.int(row[4]):np.int(row[6]),
                        np.int(row[3]):np.int(row[5]), :]

            X.append(im)
            labels.append(c)
        gt_file.close()

    # perform feature extraction
    X = _extract_feature(X, feature)

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)


    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    return (X_train, y_train), (X_test, y_test)



def _extract_feature(X, feature):
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

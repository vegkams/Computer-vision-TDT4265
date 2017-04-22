#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module that contains various classifiers"""

import cv2
import numpy as np
import os
import shutil

from abc import ABCMeta, abstractmethod
from matplotlib import pyplot as plt




class Classifier:
    """
        Abstract base class for all classifiers

        A classifier needs to implement at least two methods:
        - fit:       A method to train the classifier by fitting the model to
                     the data.
        - evaluate:  A method to test the classifier by predicting labels of
                     some test data based on the trained model.

        A classifier also needs to specify a classification strategy via 
        setting self.mode to either "one-vs-all" or "one-vs-one".
        The one-vs-all strategy involves training a single classifier per
        class, with the samples of that class as positive samples and all
        other samples as negatives.
        The one-vs-one strategy involves training a single classifier per
        class pair, with the samples of the first class as positive samples
        and the samples of the second class as negative samples.

        This class also provides method to calculate accuracy, precision,
        recall, and the confusion matrix.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test, visualize=False):
        pass

    def _accuracy(self, y_test, Y_vote):
        """Calculates accuracy

            This method calculates the accuracy based on a vector of
            ground-truth labels (y_test) and a 2D voting matrix (Y_vote) of
            size (len(y_test), num_classes).

            :param y_test: vector of ground-truth labels
            :param Y_vote: 2D voting matrix (rows=samples, cols=class votes)
            :returns: accuracy e[0,1]
        """
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        # all cases where predicted class was correct
        mask = y_hat == y_test
        return np.float32(np.count_nonzero(mask)) / len(y_test)

    def _precision(self, y_test, Y_vote):
        """Calculates precision

            This method calculates precision extended to multi-class
            classification by help of a confusion matrix.

            :param y_test: vector of ground-truth labels
            :param Y_vote: 2D voting matrix (rows=samples, cols=class votes)
            :returns: precision e[0,1]
        """
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        if self.mode == "one-vs-one":
            # need confusion matrix
            conf = self._confusion(y_test, Y_vote)

            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false positives: label is c, classifier predicted not c
                fp = np.sum(conf[:, c]) - conf[c, c]

                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
        elif self.mode == "one-vs-all":
            # consider each class separately
            prec = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false positives: label is c, classifier predicted not c
                fp = np.count_nonzero((y_test == c) * (y_hat != c))

                if tp + fp != 0:
                    prec[c] = tp * 1. / (tp + fp)
        return prec

    def _recall(self, y_test, Y_vote):
        """Calculates recall
            This method calculates recall extended to multi-class
            classification by help of a confusion matrix.

            :param y_test: vector of ground-truth labels
            :param Y_vote: 2D voting matrix (rows=samples, cols=class votes)
            :returns: recall e[0,1]
        """
        # predicted classes
        y_hat = np.argmax(Y_vote, axis=1)

        if self.mode == "one-vs-one":
            # need confusion matrix
            conf = self._confusion(y_test, Y_vote)

            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = conf[c, c]

                # false negatives: label is not c, classifier predicted c
                fn = np.sum(conf[c, :]) - conf[c, c]
                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        elif self.mode == "one-vs-all":
            # consider each class separately
            recall = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                # true positives: label is c, classifier predicted c
                tp = np.count_nonzero((y_test == c) * (y_hat == c))

                # false negatives: label is not c, classifier predicted c
                fn = np.count_nonzero((y_test != c) * (y_hat == c))

                if tp + fn != 0:
                    recall[c] = tp * 1. / (tp + fn)
        return recall

    def _confusion(self, y_test, Y_vote):
        """Calculates confusion matrix

            This method calculates the confusion matrix based on a vector of
            ground-truth labels (y-test) and a 2D voting matrix (Y_vote) of
            size (len(y_test), num_classes).
            Matrix element conf[r,c] will contain the number of samples that
            were predicted to have label r but have ground-truth label c.

            :param y_test: vector of ground-truth labels
            :param Y_vote: 2D voting matrix (rows=samples, cols=class votes)
            :returns: confusion matrix
        """
        y_hat = np.argmax(Y_vote, axis=1)
        conf = np.zeros((self.num_classes, self.num_classes)).astype(np.int32)
        for c_true in range(self.num_classes):
            # looking at all samples of a given class, c_true
            # how many were classified as c_true? how many as others?
            for c_pred in range(self.num_classes):
                y_this = np.where((y_test == c_true) * (y_hat == c_pred))
                conf[c_pred, c_true] = np.count_nonzero(y_this)
        return conf


class MultiClassSVM(Classifier):
    """
        Multi-class classification using Support Vector Machines (SVMs)

        This class implements an SVM for multi-class classification. Whereas
        some classifiers naturally permit the use of more than two classes
        (such as neural networks), SVMs are binary in nature.

        However, we can turn SVMs into multinomial classifiers using at least
        two different strategies:
        * one-vs-all: A single classifier is trained per class, with the
                      samples of that class as positives (label 1) and all
                      others as negatives (label 0).
        * one-vs-one: For k classes, k*(k-1)/2 classifiers are trained for each
                      pair of classes, with the samples of the one class as
                      positives (label 1) and samples of the other class as
                      negatives (label 0).

        Each classifier then votes for a particular class label, and the final
        decision (classification) is based on a majority vote.
    """

    def __init__(self, num_classes, mode="one-vs-all", params=None):
        """
            The constructor makes sure the correct number of classifiers is
            initialized, depending on the mode ("one-vs-all" or "one-vs-one").

            :param num_classes: The number of classes in the data.
            :param mode:        Which classification mode to use.
                                "one-vs-all": single classifier per class
                                "one-vs-one":  single classifier per class pair
                                Default: "one-vs-all"
            :param params:      SVM training parameters.
                                For now, default values are used for all SVMs.
                                Hyperparameter exploration can be achieved by
                                embedding the MultiClassSVM process flow in a
                                for-loop that classifies the data with
                                different parameter values, then pick the
                                values that yield the best accuracy.
                                Default: None
        """
        self.num_classes = num_classes
        self.mode = mode
        self.params = params or dict()

        self.writeToFile = True
        self.writeImagesToFolder = True
        self.drawRectangles = True

        self.signMapping = ["speed limit 20 (prohibitory)",
                            "speed limit 30 (prohibitory)",
                            "speed limit 50 (prohibitory)",
                            "speed limit 60 (prohibitory)",
                            "speed limit 70 (prohibitory)",
                            "speed limit 80 (prohibitory)",
                            "restriction ends 80 (other)",
                            "speed limit 100 (prohibitory)",
                            "speed limit 120 (prohibitory)",
                            "no overtaking (prohibitory)",
                            "no overtaking (trucks) (prohibitory)",
                            "priority at next intersection (danger)",
                            "priority road (other)",
                            "give way (other)",
                            "stop (other)",
                            "no traffic both ways (prohibitory)",
                            "no trucks (prohibitory)",
                            "no entry (other)",
                            "danger (danger)",
                            "bend left (danger)",
                            "bend right (danger)",
                            "bend (danger)",
                            "uneven road (danger)",
                            "slippery road (danger)",
                            "road narrows (danger)",
                            "construction (danger)",
                            "traffic signal (danger)",
                            "pedestrian crossing (danger)",
                            "school crossing (danger)",
                            "cycles crossing (danger)",
                            "snow (danger)",
                            "animals (danger)",
                            "restriction ends (other)",
                            "go right (mandatory)",
                            "go left (mandatory)",
                            "go straight (mandatory)",
                            "go right or straight (mandatory)",
                            "go left or straight (mandatory)",
                            "keep right (mandatory)",
                            "keep left (mandatory)",
                            "roundabout (mandatory)",
                            "restriction ends (overtaking) (other)",
                            "restriction ends (overtaking (trucks)) (other)"]

        # initialize correct number of classifiers
        self.classifiers = []
        if mode == "one-vs-one":
            # k classes: need k*(k-1)/2 classifiers
            for _ in range((int)(num_classes*(num_classes - 1) / 2)):
                svm = cv2.ml.SVM_create()
                svm.setType(cv2.ml.SVM_C_SVC)
                svm.setKernel(cv2.ml.SVM_LINEAR)
                svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
                self.classifiers.append(svm)
        elif mode == "one-vs-all":
            # k classes: need k classifiers
            for _ in range(num_classes):
                self.classifiers.append(cv2.ml.SVM_create())
        else:
            print("Unknown mode ", mode)

    def fit(self, X_train, y_train, params=None):
        """Fits the model to training data

            This method trains the classifier on data (X_train) using either
            the "one-vs-one" or "one-vs-all" strategy.

            :param X_train: input data (rows=samples, cols=features)
            :param y_train: vector of class labels
            :param params:  dict to specify training options for cv2.SVM.train
                            leave blank to use the parameters passed to the
                            constructor
        """
        if params is None:
            params = self.params

        if self.mode == "one-vs-one":
            svm_id = 0
            for c1 in range(self.num_classes):
                for c2 in range(c1 + 1, self.num_classes):
                    # indices where class labels are either `c1` or `c2`
                    data_id = np.where((y_train == c1) + (y_train == c2))[0]

                    # set class label to 1 where class is `c1`, else 0
                    y_train_bin = np.where(y_train[data_id] == c1, 1, 0).flatten()


                    #self.classifiers[svm_id].train(X_train[data_id, :], y_train_bin, params=self.params)
                    self.classifiers[svm_id].train(X_train[data_id, :],cv2.ml.ROW_SAMPLE ,y_train_bin)
                    svm_id += 1
        elif self.mode == "one-vs-all":
            for c in range(self.num_classes):
                # train c-th SVM on class c vs. all other classes
                # set class label to 1 where class==c, else 0
                 y_train_bin = np.where(y_train == c, 1, 0).flatten()

                # train SVM
                #self.classifiers[c].train(X_train, y_train_bin, params=self.params)
                 self.classifiers[c].train(X_train, cv2.ml.ROW_SAMPLE, y_train_bin)

    def evaluateData(self, X_test, y_test, X_order, picDict, signBorders, signCounterList, rootpath = "datasets/TrainIJCNN2013"):
        """
        :param path:  Path to pictures to be evaluated
        :return:
        """
        Y_vote = np.zeros((len(y_test), self.num_classes))
        svm_id = 0
        for c1 in range(self.num_classes):
            for c2 in range(c1 + 1, self.num_classes):
                data_id = np.where((y_test == c1) + (y_test == c2))[0]
                X_test_id = X_test[data_id, :]
                y_test_id = y_test[data_id]

                # set class label to 1 where class==c1, else 0
                # y_test_bin = np.where(y_test_id==c1,1,0).reshape(-1,1)

                # predict labels
                y_hat = self.classifiers[svm_id].predict(X_test_id)

                for i in range(len(y_hat[1])):
                    if y_hat[1][i] == 1:
                        Y_vote[data_id[i], c1] += 1
                    elif y_hat[1][i] == 0:
                        Y_vote[data_id[i], c2] += 1
                    else:
                        print("y_hat[", i, "] = ", y_hat[i])


                # we vote for c1 where y_hat is 1, and for c2 where y_hat
                # is 0 np.where serves as the inner index into the data_id
                # array, which in turn serves as index into the results
                # array
                # Y_vote[data_id[np.where(y_hat == 1)[0]], c1] += 1
                # Y_vote[data_id[np.where(y_hat == 0)[0]], c2] += 1
                svm_id += 1


        y_hat = np.argmax(Y_vote, axis=1)


        signFound = [False]*len(y_test)

        # find text to all the matches
        for i in range(len(y_test)):
            if(y_hat[i] == y_test[i]):
                picDict[X_order[i]].append(str.format('%d. '% signCounterList[i]) + self.signMapping[y_hat[i]])
                signFound[i] = True


        font = cv2.FONT_HERSHEY_SIMPLEX
        results = ""


        folderPath = 'result/'

        if self.writeImagesToFolder:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            else:
                shutil.rmtree(folderPath)
                os.makedirs(folderPath)

        for key in picDict:
            if len(picDict[key]) > 0:
                im = cv2.imread(rootpath + "/" + key)

                # write classification text to picture
                counter = 0
                for text in picDict[key]:
                    # remove index form end

                    results += key + ': ' + text + '\n'
                    counter += 1
                    cv2.putText(im, text, (10, 50*counter), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # draw borders around signs
                if self.drawRectangles:
                    for border in signBorders[key]:
                        if signFound[border[4]]:
                            cv2.rectangle(im, (border[0],border[1]), (border[2],border[3]), (0,255,0),2)
                            #cv2.putText(im, str.format('%d' % signCounterList[border[4]]),(border[0]-15,border[1]), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(im, str.format('%d' % signCounterList[border[4]]), (border[0] - 15, border[1]), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                        else:
                            cv2.rectangle(im, (border[0], border[1]), (border[2], border[3]), (0, 153, 255), 2)
                            cv2.putText(im, str.format('%d' % signCounterList[border[4]]), (border[0] - 15, border[1]),font, 0.6, (0, 153, 255), 2, cv2.LINE_AA)



                # Write images to folder
                if self.writeImagesToFolder:
                    cv2.imwrite(folderPath + key, im)

        # Write out the results of the classification to text file
        if self.writeToFile:
            file = open('result.txt', 'w+')
            file.write(results)
            file.close()

    def evaluateLive(self, X_test, signBorders, rootpath = "datasets/TrainIJCNN2013"):
        """
        :param path:  Path to pictures to be evaluated
        :return:
        """
        y_hat_all = np.array(np.zeros(len(X_test)))
        picText = [None]*len(X_test)

        for j in range(len(X_test)):
            svm_id = 0
            Y_vote = np.zeros(self.num_classes)
            for c1 in range(self.num_classes):
                for c2 in range(c1 + 1, self.num_classes):

                    #data_id = np.where((y_test == c1) + (y_test == c2))[0]
                    #X_test_id = X_test[data_id, :]

                    # set class label to 1 where class==c1, else 0
                    # y_test_bin = np.where(y_test_id==c1,1,0).reshape(-1,1)

                    # predict labels
                    #y_hat = self.classifiers[svm_id].predict(np.array([X_test]))

                    y_hat = self.classifiers[svm_id].predict(np.array([X_test[j]]))

                    for i in range(len(y_hat[1])):
                        if y_hat[1][i] == 1:
                            Y_vote[c1] += 1
                        elif y_hat[1][i] == 0:
                            Y_vote[c2] += 1
                        else:
                            print("y_hat[", i, "] = ", y_hat[i])

                    # we vote for c1 where y_hat is 1, and for c2 where y_hat
                    # is 0 np.where serves as the inner index into the data_id
                    # array, which in turn serves as index into the results
                    # array
                    # Y_vote[data_id[np.where(y_hat == 1)[0]], c1] += 1
                    # Y_vote[data_id[np.where(y_hat == 0)[0]], c2] += 1
                    svm_id += 1
            y_hat = np.argmax(Y_vote)
            y_hat_all[j] = y_hat
            picNumber = j + 1
            picText[j] = str.format('%d. %s'% (picNumber, self.signMapping[y_hat]))
        print(y_hat_all)

       # find text to the match
        font = cv2.FONT_HERSHEY_SIMPLEX
        results = ""

        folderPath = 'result/'

        if self.writeImagesToFolder:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            else:
                shutil.rmtree(folderPath)
                os.makedirs(folderPath)

            pic = "00001.ppm"
            im = cv2.imread(rootpath + "/" + pic)

            # write classification text to picture
            counter = 0
            for text in picText:


      #             results += key + ': ' + text + '\n'
                counter += 1
                cv2.putText(im, text, (10, 50*counter), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # draw borders around signs
            counter = 1
            if self.drawRectangles:
                for border in signBorders:
                    cv2.rectangle(im, (border[0],border[1]), (border[2],border[3]), (0,255,0),2)
                    #cv2.putText(im, str.format('%d' % signCounterList[border[4]]),(border[0]-15,border[1]), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(im, str.format('%d' % counter), (border[0] - 15, border[1]), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    counter += 1


      #         # Write images to folder
                if self.writeImagesToFolder:
                    cv2.imwrite(folderPath + pic, im)

       # Write out the results of the classification to text file
        if self.writeToFile:
            file = open('result.txt', 'w+')
            file.write(results)
            file.close()




    def evaluate(self, X_test, y_test, visualize=False):
        """Evaluates the model on test data

            This method evaluates the classifier's performance on test data
            (X_test) using either the "one-vs-one" or "one-vs-all" strategy.

            :param X_test:    input data (rows=samples, cols=features)
            :param y_test:    vector of class labels
            :param visualize: flag whether to plot the results (True) or not
                              (False)
            :returns: accuracy, precision, recall
        """
        # prepare Y_vote: for each sample, count how many times we voted
        # for each class
        Y_vote = np.zeros((len(y_test), self.num_classes))

        if self.mode == "one-vs-one":
            svm_id = 0
            for c1 in range(self.num_classes):
                for c2 in range(c1 + 1, self.num_classes):
                    data_id = np.where((y_test == c1) + (y_test == c2))[0]
                    X_test_id = X_test[data_id, :]
                    y_test_id = y_test[data_id]

                    # set class label to 1 where class==c1, else 0
                    # y_test_bin = np.where(y_test_id==c1,1,0).reshape(-1,1)

                    # predict labels
                    y_hat = self.classifiers[svm_id].predict(X_test_id)

                    for i in range(len(y_hat[1])):
                        if y_hat[1][i] == 1:
                            Y_vote[data_id[i], c1] += 1
                        elif y_hat[1][i] == 0:
                            Y_vote[data_id[i], c2] += 1
                        else:
                            print("y_hat[", i, "] = ", y_hat[i])

                    # we vote for c1 where y_hat is 1, and for c2 where y_hat
                    # is 0 np.where serves as the inner index into the data_id
                    # array, which in turn serves as index into the results
                    # array
                    # Y_vote[data_id[np.where(y_hat == 1)[0]], c1] += 1
                    # Y_vote[data_id[np.where(y_hat == 0)[0]], c2] += 1
                    svm_id += 1


        elif self.mode == "one-vs-all":
            for c in range(self.num_classes):
                # set class label to 1 where class==c, else 0
                # predict class labels
                # y_test_bin = np.where(y_test==c,1,0).reshape(-1,1)

                # predict labels
                y_hat = self.classifiers[c].predict_all(X_test)

                # we vote for c where y_hat is 1
                if np.any(y_hat):
                    Y_vote[np.where(y_hat == 1)[0], c] += 1

            # with this voting scheme it's possible to end up with samples
            # that have no label at all...in this case, pick a class at
            # random...
            no_label = np.where(np.sum(Y_vote, axis=1) == 0)[0]
            Y_vote[no_label, np.random.randint(self.num_classes,
                                               size=len(no_label))] = 1

        accuracy = self._accuracy(y_test, Y_vote)
        precision = self._precision(y_test, Y_vote)
        recall = self._recall(y_test, Y_vote)
        return accuracy, precision, recall

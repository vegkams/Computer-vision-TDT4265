#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt

import gtsrb
from classifiers import MultiClassSVM

plotComparisson = False

def main():
    #strategies = ['one-vs-one', 'one-vs-all']
    strategies = ['one-vs-one']
    #features = [None, 'gray', 'rgb', 'hsv', 'hog']
    features = ['hog']
    accuracy = np.zeros((2, len(features)))
    precision = np.zeros((2, len(features)))
    recall = np.zeros((2, len(features)))

    for f in range(len(features)):
        print("feature", features[f])
        (X_train, y_train), (X_test, y_test) = gtsrb.load_data(feature=features[f],test_split=0.2,seed=42)

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


            testData = gtsrb.load_test_data()
            X_test = np.squeeze(np.array(testData[0])).astype(np.float32)
            y_test = np.array(testData[1])
            MCS.evaluateData(X_test, y_test, testData[2], testData[3], testData[4], testData[5])



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




if __name__ == '__main__':
    main()

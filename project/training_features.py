import numpy as np
import cv2
import csv
from skimage.feature import hog
from skimage import color


class TrainingFeatures:
    def __init__(self):
        pass

    def extract_feature(self, X, feature):
        # Operate on images of equal sizes
        size = (32, 32)
        print("Starting feature extraction\n")

        x_data = [cv2.resize(x, size) for x in X]
        x_data = [color.rgb2gray(x) for x in x_data]


        # Normalize intensities and subtract mean
        x_data = np.array(x_data).astype(np.float32) / 255
        x_data = [x - np.mean(x) for x in x_data]

        if feature == 'hog':
            x_data = [hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for x in x_data]
        x_data = [x.flatten() for x in x_data]
        print("Returning extracted features\n")
        return x_data

    # function for reading the images
    # arguments: path to the traffic sign data, for example './GTSRB/Training'
    # returns: list of images, list of corresponding labels
    def load_data(self, rootpath="datasets/GTSRB/Final_Training/Images", cut_roi=True,
                  test_split=0.2, seed=113):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        # class prohibitory signs (circular ,white background/red border)
        classes = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16])
        X = []  # images
        labels = []  # corresponding labels
        # loop over all classes
        for c in range(len(classes)):
            prefix = rootpath + '/' + format(classes[c], '05d') + '/'  # subdirectory for class
            gtFile = open(prefix + 'GT-' + format(classes[c], '05d') + '.csv')  # annotations file
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
            next(gtReader)  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                im = cv2.imread(prefix + row[0])  # the 1st column is the filename
                if cut_roi:
                    im = im[np.int(row[4]):np.int(row[6]),
                         np.int(row[3]):np.int(row[5]), :]
                X.append(im)
                labels.append(row[7])  # the 8th column is the label
            gtFile.close()
            ''' TODO: extract feature here?'''
        X = self.extract_feature(X, 'hog')
        # Shuffle before splitting into training and validation set
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(labels)
        X_train = X[:int(len(X) * (1 - test_split))]
        y_train = labels[:int(len(X) * (1 - test_split))]
        X_test = X[int(len(X) * (1 - test_split)):]
        y_test = labels[int(len(X) * (1 - test_split)):]

        return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    ft = TrainingFeatures()
    ft.load_data()
    print("Done")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread
import cv2
import csv
from skimage.feature import hog
from skimage import color, data, exposure


img = mpimg.imread("FullIJCNN2013/00003.ppm")
image = color.rgb2gray(img)
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def load_data(rootpath="datasets",cut_roi=True):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    # class prohibitory signs (circular ,white background/red border)
    classes = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16])
    X = [] # images
    labels = [] # corresponding labels
    # loop over all classes
    for c in range(len(classes)):
        prefix = rootpath + '/' + format(classes[c], '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(classes[c], '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels

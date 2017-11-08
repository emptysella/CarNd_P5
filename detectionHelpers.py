import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import pickle

""" 1.-> Extract HOG features """
def get_hog_features(img, orient, pix_per_cell, cell_per_block):

    hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=False, feature_vector=False)

    return hog_features


""" 1.-> Extract HOG features """
def ExtractHOG(img, orient, pix_per_cell, cell_per_block):

    hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=False, feature_vector=True)

    return hog_features

""" 2.-> binned color features """
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


""" 3.-> color histogram features """
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

""" 4.-> Extract cobined features """
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        orient = 9, pix_per_cell = 8, cell_per_block = 2):
    # Create a list to append feature vectors to
    features = []
    c=0
    # Iterate through the list of images
    #for file in imgs:

    img = shuffle(imgs, random_state=42)
    for idx in range(0,5000):
        file = img[idx]
        print(c)
        c +=1
        # Read in each one by one
        imageA = mpimg.imread(file)
        image = cv2.resize(imageA, (64,64))
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Apply ExtractHOG() to get HOG features
        #Hof_feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        #HOG_feature = ExtractHOG(Hof_feature_image, orient, pix_per_cell, cell_per_block)
        HOG_feature_H = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block).ravel()
        HOG_feature_S = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block).ravel()
        HOG_feature_V = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block).ravel()

        # Append the new feature vector to the features list
        #features.append(HOG_feature[0])
        features.append(np.concatenate(( spatial_features,hist_features, HOG_feature_H, HOG_feature_S, HOG_feature_V)))

    # Return list of feature vectors
    return features


""" 5.-> Get images by classes """
def imagesByClasses(ImagesClassA,ImagesClassB ):
    cars = []
    notcars = []
    for imageNC in ImagesClassB:
        if 'image' in imageNC or 'extra' in imageNC:
            notcars.append(imageNC)


    for imageC in ImagesClassA:
        if 'micarro' in imageC:
            notcars.append(imageC)
        else:
                cars.append(imageC)
    return cars, notcars

""" 6.-> Extract features by class """
def featuresByClasses(ClassA, ClassB, cspace='RGB', spatial_size=(32, 32),
                    hist_bins=32, hist_range=(0, 256), orient = 9,
                    pix_per_cell = 8, cell_per_block = 2):

    print('ClassA')
    ClassA_features = extract_features(ClassA, cspace, spatial_size,hist_bins,
                                hist_range, orient, pix_per_cell, cell_per_block)

    ClassB_features = extract_features(ClassB, cspace, spatial_size,hist_bins,
                                hist_range, orient, pix_per_cell, cell_per_block)
    print('ClassB')
    X = np.vstack((ClassA_features, ClassB_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    with open('X_scaler_7_YCrCb_5000IMG.pkl', 'wb') as f:
        pickle.dump(X_scaler, f)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    return scaled_X

# General Imports

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

## -----
## METHODS

# Helper function to read in images from center, left and right cameras
def readx3(angles, images, batch_sample, correction, path):

    # read in images
    img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
    img_left = cv2.imread(path + batch_sample[1].split('/')[-1])
    img_right = cv2.imread(path + batch_sample[2].split('/')[-1])
    steering_center = float(batch_sample[3])

    # Apply correction on central angle for left and right image
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # Augment the dataset through flipping the image
    augmented_img_center = cv2.flip(img_center, 1)
    augmented_img_left = cv2.flip(img_left, 1)
    augmented_img_right = cv2.flip(img_right, 1)
    augmented_steering_center = (steering_center * -1.0)
    augmented_steering_left = (steering_left * -1.0)
    augmented_steering_right = (steering_right * -1.0)

    # Append
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)

    angles.append(steering_center)
    angles.append(steering_left)
    angles.append(steering_right)

    images.append(augmented_img_center)
    images.append(augmented_img_left)
    images.append(augmented_img_right)

    angles.append(augmented_steering_center)
    angles.append(augmented_steering_left)
    angles.append(augmented_steering_right)


# Define a generator function to consolidate datasets together
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # identify data source
                if (batch_sample[-1] == 0):
                    # Path for the images
                    path = '../Udacity-data/IMG/'

                    # Correction factor
                    correction = 0.2

                    # Read and append
                    readx3(angles, images, batch_sample, correction, path)

                elif (batch_sample[-1] == 1):
                    # Path for the images
                    path = '../My-Data-20190505/IMG/'

                    # Correction factor
                    correction = 0.2

                    # Read and append
                    readx3(angles, images, batch_sample, correction, path)

                elif (batch_sample[-1] == 2):
                    # Path for the images
                    path = '../My-Data-20190508/IMG/'

                    # Correction factor
                    correction = 0.2

                    # Read and append
                    readx3(angles, images, batch_sample, correction, path)

                elif (batch_sample[-1] == 3):
                    # Path for the images
                    path = '../My-Data-recovery-20190512/IMG/'

                    # Correction factor
                    correction = 0.2

                    # Read and append
                    readx3(angles, images, batch_sample, correction, path)


                else:
                    print('SAMPLE ERROR')
                    return

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

## END METHODS

## -----
## MAIN SCRIPT

# 1. Read data
# Define sample images vectors
samples1 = []
samples2 = []
samples3 = []

# Define collective vectors of samples for training and validation
train_samples = []
validation_samples = []

# Read sample  images from test runs, introducing a "flag" variable at the end of every line we read
with open('../Udacity-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line.append(0)  # Flag for Udacity Data
        samples1.append(line)

size_ud_data = len(samples1)

with open('../My-Data-20190505/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line.append(1)  # Flag for my data
        samples2.append(line)

size_my_data = len(samples2)

with open('../My-Data-20190508/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line.append(2)  # Flag for my data - set 2
        samples3.append(line)

size_my_data2 = len(samples3)

# Split in testing and validation samples
train_samples1, validation_samples1 = train_test_split(samples1, test_size=0.2)
train_samples2, validation_samples2 = train_test_split(samples2, test_size=0.2)
train_samples3, validation_samples3 = train_test_split(samples3, test_size=0.2)

# Append vectors together
train_samples = train_samples1 + train_samples2 + train_samples3
validation_samples = validation_samples1 + validation_samples2 + validation_samples3

# Uncomment to print info
# print('Size of my set : ', size_my_data)
# print('Size of my reverse set : ', size_my_data2)
# print('Size of reference ud set : ', size_ud_data)
# print('Size of data set : ', size_my_data + size_my_data2 + size_ud_data)

# print('Size of training set : ', len(train_samples))
# print('Size of validation set : ', len(validation_samples))

# 2. Pair a generator function against test/validation samples
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# 3. Imports for the network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# 4. Define and train the network
model = Sequential()

# Cropping
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# Nvidia network structure
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Define optimizer
model.compile(loss='mse', optimizer='adam')

# Train the model
# NOTE: samples_per_epoch gets multiplied because of augmentation/reading 3 images
model.fit_generator(train_generator, samples_per_epoch=(6 * len(train_samples)), 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

# Save model
model.save('model_h5')

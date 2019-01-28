import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

data_path = "mydata/"
def prepare_image(source_path):
    path = data_path + "IMG/"
    file_name = source_path.split('/')[-1]
    return cv2.cvtColor(cv2.imread(path + file_name), cv2.COLOR_BGR2RGB)

def read_samples():
    lines = []
    with open(data_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines[1:] 

def generate_data(samples):
    images = []
    angles = []
    for sample in samples:
        steering_center = float(sample[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = prepare_image(sample[0])
        img_left = prepare_image(sample[1])
        img_right = prepare_image(sample[2])

        # add images and angles to data set
        images.extend((img_center, img_left, img_right))
        angles.extend((steering_center, steering_left, steering_right))

        # augment data set
        images.extend((cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1)))
        angles.extend((steering_center * -1.0, steering_left * -1.0, steering_right * -1.0))

    X_train = np.array(images)
    y_train = np.array(angles)
    return X_train, y_train

def train():
    samples = read_samples()
    X_train, y_train = generate_data(samples)

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    model.save('model.h5')
    exit()
    
train()
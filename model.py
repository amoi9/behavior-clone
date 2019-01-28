import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Lambda, Dense, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

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

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                img_center = prepare_image(batch_sample[0])
                img_left = prepare_image(batch_sample[1])
                img_right = prepare_image(batch_sample[2])

                # add images and angles to data set
                images.extend((img_center, img_left, img_right))
                angles.extend((steering_center, steering_left, steering_right))

                # augment data set
                images.extend((cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1)))
                angles.extend((steering_center * -1.0, steering_left * -1.0, steering_right * -1.0))

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def train():
    samples = read_samples()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

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
    model.fit_generator(train_generator, samples_per_epoch=
                len(train_samples), validation_data=validation_generator,
                nb_val_samples=len(validation_samples), nb_epoch=3)

    model.save('model.h5')
    exit()
    
train()
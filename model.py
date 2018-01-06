import csv
import cv2
import numpy as np
import os
import json
import h5py
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# with open('/Users/strikefreedom2017/Desktop/Data/driving_log.csv') as csvfile:

steering_correction = 0.2
def get_imgPath_measurementValue( csv_filePath ):

    images_path = []
    steering_angles = []

    with open( csv_filePath ) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:

            img_center_filePath = line[0]
            img_left_filePath   = line[1]
            img_right_filePath  = line[2]

            steering_angle_center = float(line[3])
            steering_angle_left   = steering_angle_center + steering_correction
            steering_angle_right  = steering_angle_center - steering_correction

            images_path.append(img_center_filePath)
            images_path.append(img_left_filePath)
            images_path.append(img_right_filePath)

            steering_angles.append(steering_angle_center)
            steering_angles.append(steering_angle_left)
            steering_angles.append(steering_angle_right)

    return images_path, steering_angles


# -----------------------------------------------------------------------------------------------------
def batch_gen( data, batch_sample_size=32 ):

    n = len(data)
    while 1:
        data_shuffle = shuffle(data)

        for ii in range(0, n, batch_sample_size):

            ind_start = ii
            ind_end   = ii + batch_sample_size
            cropped_data = data[ind_start : ind_end]

            images = []
            steering_angles = []

            for image_path, steering_angle in cropped_data:

                image_org  = cv2.imread(image_path)
                image      = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
                image_flip = cv2.flip(image,-1)

                images.append(image)
                images.append(image_flip)

                steering_angles.append(steering_angle)
                steering_angles.append(steering_angle*-1.0)

            X_array = np.array(images)
            Y_array = np.array(steering_angles)

            yield shuffle(X_array, Y_array)



def Nvidia_Model():

    model = Sequential()

    # Preprocessing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((45, 20), (0, 0))))

    # --------------------------------------------------------------------
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2) ) )
    model.add(Activation('relu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))

    # --------------------------------------------------------------------
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # --------------------------------------------------------------------
    model.compile(loss='mse', optimizer='adam')

    return model



# -----------------------------------------------------------------------------------------------------

images_path, steering_angles = get_imgPath_measurementValue('/Users/strikefreedom2017/Desktop/Data/driving_log.csv')
train_XY, valid_XY = train_test_split( list(zip(images_path, steering_angles)), test_size=0.2)

model = Nvidia_Model()
history = model.fit_generator( batch_gen(train_XY, batch_sample_size=32), samples_per_epoch=len(train_XY), \
                               validation_data=batch_gen(valid_XY, batch_sample_size=32), nb_val_samples=len(valid_XY), \
                               nb_epoch=3)

model.save('model3.h5')




Training_loss   = history.history['loss']
Validation_loss = history.history['val_loss']
print('Training_Loss = {}'.format(Training_loss) )
print('Validation Loss = {}'.format(Validation_loss))

plt.figure()
plt.plot(Training_loss, '-o', label='Training')
plt.plot(Validation_loss, '-x', label='Validation')
plt.xlabel('Epoch [-]')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

print("Completed :)")
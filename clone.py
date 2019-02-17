import csv
import cv2
import numpy as np
from scipy import ndimage

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/' + batch_sample[0].split('\\')[-1]
                center_image = ndimage.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement)
                augmented_measurements.append(-measurement)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Activation, Dropout, Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,
          input_shape=(row, col, ch),
          output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70, 26), (0, 0))))
model.add(Convolution2D(6, (5, 5), border_mode='same', activation='relu'))
model.add(MaxPooling2D(4, 4))
model.add(Convolution2D(16, (5, 5), border_mode='same', activation='relu'))
model.add(MaxPooling2D(2, 4))
model.add(Convolution2D(33, (3, 3), border_mode='same', activation='relu'))
model.add(MaxPooling2D(1, 2))
model.add(Flatten())
model.add(Dense(190))
model.add(Dropout(0.75))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples),
                    epochs=3,
                    verbose=1)



# images = []
# measurements = []
# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('\\')[-1]
#     current_path = '../data/IMG/{0}'.format(filename)
#     image = ndimage.imread(current_path)
#     images.append(image)
    
#     measurement = float(line[3])
#     measurements.append(measurement)
    
# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_images.append(cv2.flip(image, 1))
#     augmented_measurements.append(measurement)
#     augmented_measurements.append(-measurement)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Lambda, Cropping2D, Activation, Dropout, Convolution2D
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPooling2D

# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((70, 26), (0, 0))))
# model.add(Convolution2D(6, (5, 5), border_mode='same', activation='relu'))
# model.add(MaxPooling2D(4, 4))
# model.add(Convolution2D(16, (5, 5), border_mode='same', activation='relu'))
# model.add(MaxPooling2D(2, 4))
# model.add(Convolution2D(33, (3, 3), border_mode='same', activation='relu'))
# model.add(MaxPooling2D(1, 2))
# model.add(Flatten())
# model.add(Dense(190))
# model.add(Dropout(0.75))
# model.add(Activation('relu'))
# model.add(Dense(84))
# model.add(Activation('relu'))
# model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
exit()
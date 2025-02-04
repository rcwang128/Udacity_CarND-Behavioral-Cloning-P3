import numpy as np
import csv
import cv2
import tensorflow as tf
from random import shuffle
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split 
import sklearn

# This is the data path where all training data are stored
data_path = 'myData0'

# List to store all lines from csv file
samples = []

# Open csv file to grab all lines for processing
with open(data_path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
csvfile.close()

del samples[0]

# Split samples and use 20 percent for test data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Enable generator with batch size 32
def generator(samples, batch_size=32):
    n_samples = len(samples)
    print(n_samples)
    while 1:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:
                file_name = data_path + '/IMG/' + batch_sample[0].split('/')[-1]
                file_name1 = data_path + '/IMG/' + batch_sample[1].split('/')[-1]
                file_name2 = data_path + '/IMG/' + batch_sample[2].split('/')[-1]
                # Grab all center, left and right  images
                center_image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
                images.append(center_image)
                left_image = cv2.cvtColor(cv2.imread(file_name1), cv2.COLOR_BGR2RGB)
                images.append(left_image)
                right_image = cv2.cvtColor(cv2.imread(file_name2), cv2.COLOR_BGR2RGB)
                images.append(right_image)
                # Grab corresponding steering angle data and add +/- 0.2 correction for left/right turns
                center_steering = float(batch_sample[3])
                steering.append(center_steering)
                steering_correction = 0.2
                left_steering = center_steering + steering_correction
                steering.append(left_steering)
                right_steering = center_steering - steering_correction
                steering.append(right_steering)
                # Flip all image and steeringi angle to balance data
                images.append(cv2.flip(center_image, 1))
                steering.append(-center_steering)
            
            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Apply Nvidia's model for training
model = Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3))) # Data normalization
model.add(Cropping2D(cropping=((65,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
adam = optimizers.Adam(lr=0.0001) # Lower down training rate for Adam optimizer
model.compile(loss='mse', optimizer=adam, metrics=['mae','acc'])

train_steps = math.ceil(len(train_samples)/batch_size)
val_steps = math.ceil(len(validation_samples)/batch_size)
model.fit_generator(train_generator, steps_per_epoch=train_steps, validation_data=validation_generator, validation_steps=val_steps, nb_epoch=5)

# Save the model
model.save('model2.h5')


import numpy as np
import csv
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split 
import sklearn


data_path = 'data_2laps'

samples = []

with open(data_path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
csvfile.close()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

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
                center_image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
                center_steering = float(batch_sample[3])
                images.append(center_image)
                steering.append(center_steering)
                
                # Flip image/measurement
                images.append(cv2.flip(center_image, 1))
                steering.append(-center_steering)
            
            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)

    print(len(X_train)); print(len(y_train))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(70,320,3)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model2.h5')


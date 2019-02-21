import numpy as np
import csv
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Flatten, Dense, Dropout

data_path = 'myData1'

lines = []
images = []
steering = []


with open(data_path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Delet first title line
del lines[0]
#print(lines[0])

for line in lines:
    file_name = line[0].split('/')[-1]
    #print (file_name)
    image_path = data_path + '/IMG/' + file_name
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    images.append(image)
    steering.append(float(line[3]))
    # Flip the image/measurement
    images.append(cv2.flip(image, 1))
    steering.append(-float(line[3]))

X_train = np.array(images)
y_train = np.array(steering)

model = Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')


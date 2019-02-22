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
    image_center = data_path + '/IMG/' + line[0].split('/')[-1]
    image_left = data_path + '/IMG/' + line[1].split('/')[-1]
    image_right = data_path + '/IMG/' + line[2].split('/')[-1]

    img_center = cv2.cvtColor(cv2.imread(image_center), cv2.COLOR_BGR2RGB) 
    images.append(img_center)
    img_left = cv2.cvtColor(cv2.imread(image_left), cv2.COLOR_BGR2RGB) 
    images.append(img_left)
    img_right = cv2.cvtColor(cv2.imread(image_right), cv2.COLOR_BGR2RGB) 
    images.append(img_right)

    correction = 0.2
    steering_center = float(line[3])
    steering.append(steering_center)
    steering_left = steering_center + correction 
    steering.append(steering_left)
    steering_right = steering_center - correction
    steering.append(steering_right)

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


import numpy as np
import csv
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense

lines = []
images = []
steering = []


with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Delet first title line
del lines[0]
#print(lines[0])

for line in lines:
    file_name = line[0].split('/')[-1]
    #print (file_name)
    image_path = 'data/IMG' + file_name
    image = cv2.imread(image_path)
    images.append(image)
    steering.append(line[3])

#print(steering[51])
X_train = np.array(images)
y_train = np.array(steering)


model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')


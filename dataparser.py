import numpy as np
import csv
import cv2
import random
from matplotlib import pyplot as plt

data_path = 'data'

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

    #images.extend(img_center, img_left, img_right)
    #steering.extend(steering_center, steering_left, steering_right)
    # Flip the image/measurement
    #images.append(cv2.flip(image,1))
    #steering.append(-float(line[3]))

print(len(images))
print(len(steering))

index = random.randint(0, len(images))
img = images[index].squeeze()
plt.imshow(img)

# Display
hist, bins = np.histogram(steering, bins=50)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(8,6))
plt.bar(center, hist, width=width)
plt.show()

import numpy as np
import csv
import cv2
import random
from matplotlib import pyplot as plt

data_path = 'myData0'
#data_path2 = 'data_2laps'

lines = []
images = []
steering = []


with open(data_path + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
csvfile.close()
del lines[0]
sample1 = len(lines)
print(sample1)
'''
with open(data_path2 + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
csvfile.close()
sample2 = len(lines)
print(sample2)
'''
i = 0
for line in lines:
    if i < sample1:
        image_center = data_path + '/IMG/' + line[0].split('/')[-1]
        image_left = data_path + '/IMG/' + line[1].split('/')[-1]
        image_right = data_path + '/IMG/' + line[2].split('/')[-1]
    elif i < sample2:
        image_center = data_path2 + '/IMG/' + line[0].split('/')[-1]
        image_left = data_path2 + '/IMG/' + line[1].split('/')[-1]
        image_right = data_path2 + '/IMG/' + line[2].split('/')[-1]

    img_center = cv2.cvtColor(cv2.imread(image_center), cv2.COLOR_BGR2RGB) 
    images.append(img_center)
    img_left = cv2.cvtColor(cv2.imread(image_left), cv2.COLOR_BGR2RGB) 
    images.append(img_left)
    img_right = cv2.cvtColor(cv2.imread(image_right), cv2.COLOR_BGR2RGB) 
    images.append(img_right)

    steering_center = float(line[3])
    steering.append(steering_center)
    correction = 0.2
    steering_left = steering_center + correction 
    steering.append(steering_left)
    steering_right = steering_center - correction
    steering.append(steering_right)
   
    # Flip the image/measurement
    images.append(cv2.flip(img_center,1))
    steering.append(-steering_center)
    i += 1

print(len(images))
print(len(steering))

f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,8))
ax1.imshow(images[60])
ax1.set_title('Center', fontsize=20)
ax2.imshow(images[61])
ax2.set_title('Left', fontsize=20)
ax3.imshow(images[62])
ax3.set_title('Right', fontsize=20)
plt.show()

images_new = []
for n in range(len(images)):
    img = images[n].squeeze()
    img = img[65:135, 0:320]
    images_new.append(img)
    
index = random.randint(0, len(images))
#img = images[index]
#print(img.shape)
#img = img[65:135, 0:320]
print(images_new[index].shape)
f, (ax1,ax2) = plt.subplots(1,2,figsize=(20,8))
ax1.imshow(images_new[index])
ax2.imshow(cv2.flip(images_new[index], 1))
plt.show()


# Display
hist, bins = np.histogram(steering, bins=50)
width = 0.8 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(8,6))
plt.bar(center, hist, width=width)
plt.show()


import numpy as np
import csv
import cv2
from matplotlib import pyplot as plt

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

print(len(images))
print(len(steering))

plt.hist(steering, bins=50, linewidth=0.2)
plt.show()

# **Behavioral Cloning** 

## Harry Wang
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

####  Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
The full project can be found at
https://github.com/rcwang128/Udacity_CarND-Behavioral-Cloning-P3

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network 
* drive.py for driving the car in autonomous mode
* video folder containing a clip of the car driving autonomously in simulator
* writeup_Behavioral_Cloning.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the (the first) track by executing
```sh
python drive.py model.h5
```

Two video clips can be found under video folder, which record how the car is performing autonomous driving based on my trained model.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used a convolutional neural network (CNN) model to train the image data referencing an NVIDIA paper "End to End Learning for Self-Driving Cars". This network consists of 9 layers, including 5 convolutional layers and 3 fully connected layers. The input image is kept as RGB888 data type (i.e. 3 channels) but cropped with 70 lines only keeping only road pixels. 

Below picture shows an overview of original model architecture from the paper. Minor changes are applied in this project and will be discussed in following sections.

![nvidia_cnn.png](https://github.com/rcwang128/Udacity_CarND-Behavioral-Cloning-P3/blob/master/examples/nvidia_cnn.png?raw=true)

#### 2. Attempts to reduce overfitting in the model

Additional dropout layers are added after each fully connected layer in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with fine tuned learning rate of 0.0001, which gives out the best performance.

```python
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam, metrics=['mae','acc'])
```

#### 4. Appropriate training data

I used a combination of center images, left and right images to train the model. Additionally, center images are being flipped horizontally again so that the data are more balanced.



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a CNN model similar to the one in NVIDIA's paper.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model by adding 3 additional dropout layers after each fully connected layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few sharp turns where the vehicle ran off the track. To improve the driving behavior in these cases, I recorded more driving data especially focusing on turns and recoveries from off-road. More data were bing collected including using left/right camera images as well as flipped images. I also went back tuning hyper-parameters to tune the neural network for example learning rate and batch size etc.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Two video clips can be found under video folder.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes. Here is a visualization of the architecture.

|           Layer           |           Description            |
| :-----------------------: | :------------------------------: |
|           Input           |    160x320x3 normalized image    |
|         Cropping          |             70x320x3             |
|    Convolution layer 1    | 2x2 stride, 5x5 kernel, 24 depth |
|       Activation 1        |               ReLU               |
|    Convolution layer 2    | 2x2 stride, 5x5 kernel, 36 depth |
|       Activation 2        |               ReLU               |
|    Convolution layer 3    | 2x2 stride, 5x5 kernel, 48 depth |
|       Activation 3        |               ReLU               |
|    Convolution layer 4    |       3x3 kernel, 64 depth       |
|       Activation 4        |               ReLU               |
|    Convolution layer 5    |       3x3 kernel, 64 depth       |
|       Activation 5        |               ReLU               |
|       Flatten layer       |             Flatten              |
|  Fully connected layer 6  |            100 depth             |
|       Activation 6        |               ReLU               |
|       Dropout layer       |       Keep probability 0.5       |
|  Fully connected layer 7  |             50 depth             |
|       Activation 7        |               ReLU               |
|       Dropout layer       |       Keep probability 0.5       |
|  Fully connected layer 8  |             10 depth             |
|       Activation 8        |               ReLU               |
|       Dropout layer       |       Keep probability 0.5       |
| Fully connected out layer |            outputs 1             |

And code implementation on Keras with Tensorflow on backend is showing below.

```python
model = Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I started with the example data that was collected and provided by Udacity, which shows decent number of data sets with good balance. However, I realized that those data was not good enough for me to complete a successful track lap after several attempts. The vehicle would go off road frequently on some sharp turns.

I then recorded additional laps focusing more on turning, including a reverse lap and several recoveries from the left/right sides of the road back to center. The total number of data set collected is 11,510. After couple more trials, I noticed that the vehicle tends to perform better on left turns than the right turn (which is actually the only one on first track). More data is needed as well as data augmentation. I then applied all three front facing camera images i.e. center, left and right into my data set with an correction of 0.2 on steering angles. Additionally, the center image is being flipped again for better balance. Overall data set now is 11,510 * 3 + 11,510 = 46,040, which should be good for my training model. Below shows an example of all three front facing camera images that are used for training.

![camera_images.png](https://github.com/rcwang128/Udacity_CarND-Behavioral-Cloning-P3/blob/master/examples/camera_images.png?raw=true)

By removing unnecessary lines (top 65 and bottom 25), images were cropped as (70, 320) before fed into training model. Therefore the CNN model can focus more on the road in front. A cropped image along with its flipped version is showing below.

![flipped_images.png](https://github.com/rcwang128/Udacity_CarND-Behavioral-Cloning-P3/blob/master/examples/flipped_images.png?raw=true)

Since most road is straight, the data sets contain mostly 0 steering angles, followed by +/- 0.2 ones which are the correction added based on right/left camera images. Below is the histogram of steering angle distribution in my data set.

![hist_data.png](https://github.com/rcwang128/Udacity_CarND-Behavioral-Cloning-P3/blob/master/examples/hist_data.png?raw=true)

I used generator to deal with this large amount of training data for better efficiency. And data was divided into 80% training set and 20% testing set. All data sets were being shuffled to reduce variance and avoid overfit. 

The results turned out well after several back-and-forth attempts.



### Conclusion and Discussion

This project is so far the most difficult yet interesting one to me. I've spent tens of hours on collecting data, tuning neural network, validating performance and analyzing. It was frustrating at beginning when none of the trained model was working. The vehicle went off road easily especially on the first sharp turn after bridge. After so many attempts, a decent training mode was defined without overfitting. And every thing became making sense. When I only fed center and right camera image (with correction on steering angle) data, the vehicle would make better decision on left turns but not right turn. When all three camera images were applied, and after some tuning on parameters, my vehicle could finally drive through entire track without going off-road. 

However, there's definitely a lot places can be improved. The data can be preprocessed and augmented better so that I won't need this many training sets. If time permits, I would also like to collect and train the second track, which I believe is much more fun to explore.
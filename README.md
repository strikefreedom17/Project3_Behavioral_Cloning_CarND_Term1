# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Pics/img_left_center_right.png "Img_car_view"
[image2]: ./Pics/Nvidia_Architecture.png "Nvidia_Architecture"
[image3]: ./Pics/Model_Summary.png "Model_Summary"
[image4]: ./Pics/Loss.png "Loss"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 showing the result of autonomous driving mode, viewing from a center camera
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I utilized the Nvidia convolutional neural networks according to this paper: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. The details of the model architecture is given by: ![Nvidia CNN][image2]

In Keras, you can see the model summary as model.summary() as well: ![Model_Summary][image3]

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 125-131). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116 inside Nvidia_Model function). 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road shown below ![Images from left, center, right camera][image1]

Note that I used the training data of 2 laps. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start by the basic simple model first. Once all the code is running properly, then I start to fine tuning the model. I first follow the basic 2-3 CNN layers shown in the lesson video. 
After I did some research, I found that the Nvidia model is one of the candidate model. I did try to implement the Nvidia model and see significant improvement of the drive result. 

Before passing the images to the CNN model, I apply the normalization and cropping the image described in the line 88-89 in model.py in order to improve the training loss. 

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation (20%) set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I applied some change at the activation from ELU to RELU and did collect the new data set which included more severe steering correction to the center of the lane, particulalry at the sharp corners.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 83-119) is derived from the Nvidia CNN model described before.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving.

The left and right steering correction is given by 0.2 (line 19 in model.py)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase the number of image samples by double, similar to the driving backward to the track (line 70-74 in model.py)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 in this case. I used an adam optimizer so that manually training the learning rate wasn't necessary.


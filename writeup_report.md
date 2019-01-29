# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 for driving in autonomous mode 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline 
I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 63, 65, 67, 69, 70), and the data is normalized in 
the model using a Keras lambda layer (code line 61). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 64, 66, and 68). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 77). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  I only used the training data from the 
csv file log.I used a combination of center lane driving, recovering from the left and right sides of the road, and 
augmented the data by flipping the images.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate 
because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation
set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the 
validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has a 0.2 dropout rate after the first three convolutional layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots 
where the vehicle fell off the track, i.e. after passing the bridge there is a sharp turn. To improve the driving behavior 
in these cases, I augmented the dataset by adding the flipped images and using left and right images too, and I created 
adjusted steering measurements for the side camera images 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers 
and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I didn't create additional training set than provided. I preprocessed this data by converting images to RGB,
and normalizing the images. 

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under 
fitting. The ideal number of epochs was 5 as evidenced by the validation loss increases with more epochs which indicates
the model may be overfit to the training set, and validation loss keeps decreasing until 5. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

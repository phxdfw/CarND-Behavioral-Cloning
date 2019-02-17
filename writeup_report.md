# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
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

My model consists of a convolution neural network with 3 convolutional layers wth 5x5, 5x5, and 3x3 filter sizes and depths 6, 16, and 33 (model.py lines 62-67) 

The model includes RELU layers to introduce nonlinearity (code line 62, 64, 66), and the data is normalized in the model using a Keras lambda layer (code line 58). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 70). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 13). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Totally 6 loops of driving are recorded in the train/validation data. 4 of them are in the default counter-clockwise direction and the other 2 loops are in the opposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was a CNN, i.e. to alternatively use convolutional layer, activation layer, pooling layer for a few times, and then use fully connected layers and dropout layers.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it worked well in my last project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that a dropping out layer is added.

Then I tuned the value of dropout probability. I found 0.75 gave me a relatively good result.

Next, I flipped the data images to double its size. More data helps me improving accuracies in both training and validation sets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially at turning points. I found my training data is mostly telling how to control the car with road curve when the car is placed in the middle of the road. To improve the driving behavior in these cases, I collected more data by not only driving in the middle of the road, but gradually make the car at one side of the lane and then record how to "fix" the error to guide the car back to the middle of the lane. Otherwise the CNN did not learn well once the car accidently went off the road center. With more data and different driving styles, the training and validation result gets improved and the autonomous car could finish the loop staying in the road center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 57-74) consisted of a convolution neural network with the following layers and layer sizes:

1. convolutional layer with 5x5 filter and depth 6
2. relu layer
3. max pooling layer (4x4)
4. convolutional layer with 5x5 filter and depth 16
5. relu layer
6. max pooling layer (2x4)
7. convolutional layer with 3x3 filter and depth 33
8. relu layer
9. max pooling layer (1x2)
10. fully connected layer (190 neurons)
11. dropout layer (keeping prob is 0.75)
12. relu layer
13. fully connected layer (84 neurons)
14. relu layer

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane driving](/examples/center_2018_09_01_01_17_43_263.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from bad positions. These images show what a recovery looks like starting from the right side of the lane back to the middle of the lane:

![before](/examples/center_2018_09_05_13_13_47_962.jpg)
![later](/examples/center_2018_09_05_13_13_48_910.jpg)
![final](/examples/center_2018_09_05_13_13_50_206.jpg)

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images thinking that this would generate doubled samples with no extra cost. For example, here is an image that has then been flipped:

![original image](/examples/center_2018_09_01_01_17_35_743.jpg)
![flipped image](/examples/flip.png)


After the collection process, I had 23455 number of data points. I then preprocessed this data by normalization in the lambda layer.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5, as evidenced by the validation error does not decrease further. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I have also tried two versions of clone.py: one uses generator while the other does not. The one with generator is much slower in run time, but saves space. So my understanding is that the generator needs only to be turned on when the dataset is so large that the available memory is not enough.


## Behavioral Cloning Project


The goal of this project is the design, training and testing of a Convolutional Neural Network model built using [Keras](https://keras.io/).

The main intent is to implement behavioral cloning from a human driver, hence the input data will be generated through a computer simulator. The user will have to drive the vehicle along a specified track through steering and acceleration/deceleration commands. During this process, images will be saved emulating what would be captured by 3 cameras mounted on the front of the car (left/middle/right). These images will then be used to train an appriate CNN, and the results of the training will be saved as a model to be connected to the same simulator used for the data collection. The model will output the steering angle based on the images captured, while the speed is kept constant through a PID controller.

The Python script containing the Network and the training steps is [model.py](./model.py), and it will be analyzed here in the following. This Git repo contains also another script ([drive.py](./drive.py)) that was provided by the Udacity team and was used to connect the model to the simulator to allow autonomous driving.

Here below I will give detail of the experience and design in independent sections dedicated to Data Collection, Model Design and Training and Autonomus Driving Results.

---
## Data Collection

The data collection phase has probably been the most demanding part of this project. As it will be detailed in the next section, for the actual network design I decided to rely on something relatively consolidated; on the other hand, putting together an effective data set, or better a combination of datasets, took several different attemps and quite a bit of trial-and-error.

Finally, the best results have been obtained making use of 3 data sets:

* A "clean" driving of the track, operated by me;
* A "reverse" driving of the track, in which, after a U-turn at the very beginning I've driven the car in the opposite of the normal direction;
* A data set provided as reference directly by Udacity.

Despite the relative simplicity of these scenarios, their combination succeeded in producing a valid round of the vehicle along the track.

The datasets have been however further enriched as decribed in the `readx3()` helper function provided as part of [model.py](./model.py) script (lines 14-49). In particular, for all the data sets the images from all the cameras are used: the recorder steering angle is used as a label for the image coming from the middle camera, while the one to be used the left/right camera is obtained from the recorded angle +/- a fixed (configurable) correction value. Furthermore every image has been vertically flipped and the relative steering angle changed in sign, so to emulate a run of the track in reverse.

The three datasets are firstly parsed making use of the `driving_log.csv` logfile that gets generated with every run ([model.py](./model.py), lines xx-yy). Lines in the logfile are ordered cronologically, and every one of them looks like this:

*INSERT LOGILE EXTRACT*

It contains the identifiers for the image files coming from the 3 cameras as well as the steering angle at the moment of the capture. On top of these information I decided append a "flag", i.e. an identifier for each dataset (0/1/2), that gets used when accessing the actual images.

The content of the datasets is split in Train/Validation in an 80/20 percentage using the `train_test_split` function imported from `sklearn` ([model.py](./model.py), lines xx-yy)

The three datasets are then collected together using the `generator()` helper function ([model.py](./model.py), line xx-yy). A _generator_ is a [specific kind of function](https://wiki.python.org/moin/Generators) allowed by the Python language, that can be declared as iterator, i.e. used in a for loop. They are characterized by the usage of the `yield` keyword, and can be used when a piece of code has to access elements from a list without loading the full list in memory. This is helpful in case of lists with a heavy footprint, like the one that we are considering for this code.
In this case the `generator()` helper will return, every time, a _shuffled_ batch of 32 images/labels taken from the global dataset.

## Model Design and Training

The design of the Network implemented here is based on the Nvidia solution presented in the Udacity class. After cropping and normalizing steps ([model.py](./model.py), lines xx-yy), the CNN layers can be described as it follows:


| Layer         		|     Description	        					|      Output|
|:---------------------:|:---------------------------------------------:|:---------------------:|
|Input    | Cropped Normalized Image | 32x32x3 RGB Image |
|Convolution    | 5x5, Depth 24  | 28x28x24 |
|Convolution    | 5x5, Depth 36  | 24x24x36 |
|Convolution    | 5x5, Depth 48  | 20x20x48 |
|Convolution    | 3x3, Depth 64  | 16x16x64 |
|Convolution    | 3x3, Depth 64  | 16x16x64 |
|Flatten    | Transition from convolutional to dense  | 4480 |
|Dense    | Depth 100  | 100 |
|Dense    | Depth 50  | 50 |
|Dense    | Depth 10  | 10 |
|Dense    | Depth 1  | 1 (Final classifier) |


---
The input  model will output a steering angle to an autonomous vehicle. 

/ steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

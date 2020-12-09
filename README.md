# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This Project is submitted as part of the Udacity Self-Driving Car Nanodegree.

For it, the goal is to train, validate and test a Neural Network model using Keras. The model will output a steering angle to an autonomous vehicle. 
The intent is to clone driving behavior, hence the training data set for the model will be created by using a simulator where an operator can steer a car around a track for data collection. The same simulator will then be fed with the output of the trained model, to operate autonomously.

To complete the project, few files are are submitted as part of this Git repo: 

1. An [annotated writeup](./Behavioral_cloning_writeup.md) describing the fundamental aspects and limitations of the solution implemented;
2. A [Python script](./model.py), that is used to define and train the network;
3. A [Model file](./model.h5), output of the training process;
4. A [Python script](./drive.py), (provided by Udacity) that is used to exercise the model autonomously;
5. A [Video](./video.mp4), obtaining through a screen capturing of the simulator while driving autonomously.

**NOTE**: The video is also availabe on [YouTube](https://youtu.be/ZOKTThWdZMo)

Dependencies:
---
In order to run the code provided you will need to properly set up your environment. The refence for this is provided through the Udacity Starter Kit available [here](https://github.com/udacity/CarND-Term1-Starter-Kit).

The simulator can be downloaded from the classroom. Together with the simulator, also sample data that can be used to train the model were provided in the classroom.

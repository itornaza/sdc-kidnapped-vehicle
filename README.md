# Kidnapped vehicle

## Introduction
Our self-driving car has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

This project implements a 2 dimensional particle filter in C++. The particle filter is given a map and some initial localization information (analogous to what a GPS would provide). At each time step the filter also gets observation and control data.

The project's rubric can be found [here](https://review.udacity.com/#!/rubrics/747/view)

The simulation video on [youtube](https://youtu.be/g9WeSvtYaHo)

[//]: # (Image References)
[image1]: ./images/particle_filter.png "Particle filter simulator output"
[image2]: ./images/algorith_flowchart.png "Algorith flowchart"
[image3]: ./images/prediction_equations.png "Prediction step equations"
[image4]: ./images/transformation_equations.png "Coordinates transformation equations"
[image5]: ./images/weights_equation.png "weights update equations"

![alt text][image1]

## Particle filter flowchart

The following flowchart summarizes the particle filter algorithm to achieve the vehicle's localization.

![alt text][image2]

Image source: Udacity

The filter predicts the location of the self driving car from the velocity, yaw rate and time elapsed using the following equations:

![alt text][image3]

The observations of the self driving car sensors are taken in the vehicle's coordinate system. In order to correlate the sensors observations with the actual landmarks of the map, we have to convert the coordinates of the observations from vehicle coordinates to map coordinates.
This is performed with the homogeneous transformation using the following equations:

![alt text][image4]

The particles final weight is calculated as the product of each observations Multivariant-Gaussian probability.

The weights are updated for each particle landmark observation using the following equation:

![alt text][image5]

The mean of the Multivariate-Gaussian equation is the observation's associated landmark position and is evaluated at the point of the transformed observation's position.

## Installation
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and intall uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

## Important Dependencies

* cmake >= 3.5
    * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
    * Linux: make is installed by default on most Linux distros
    * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
    * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
    * Linux: gcc / g++ is installed by default on most Linux distros
    * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
    * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. Clone this repo
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
* On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./particle_filter`

```
$ mkdir build && cd build
$ cmake .. && make
$ ./particle_filter
```

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

## Data flow between the program and the Simulator

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.

**INPUT**: values provided by the simulator to the c++ program

// sense noisy position data from the simulator

["sense_x"] 

["sense_y"] 

["sense_theta"] 

// get the previous velocity and yaw rate to predict the particle's transitioned state

["previous_velocity"]

["previous_yawrate"]

// receive noisy observation data from the simulator, in a respective list of x/y values

["sense_observations_x"] 

["sense_observations_y"] 


**OUTPUT**: values provided by the c++ program to the simulator

// best particle values used for calculating the error evaluation

["best_particle_x"]

["best_particle_y"]

["best_particle_theta"] 

//Optional message data used for debugging particle's sensing and associations

// for respective (x,y) sensed positions ID label 

["best_particle_associations"]

// for respective (x,y) sensed positions

["best_particle_sense_x"] <= list of sensed x positions

["best_particle_sense_y"] <= list of sensed y positions

## More on inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory. 

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

### All other data the simulator provides, such as observations and controls.

> * Map data provided by 3D Mapping Solutions GmbH.



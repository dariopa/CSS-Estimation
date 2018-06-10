This repository contains code (python) to estimate the Camera Spectral Sensitivity functions from RGB images in the wild. 
Author:
- Dario Panzuto ([email](mailto:dariopa@ethz.ch))

## Dataset
The raw hyperspectral images that have been used for this project were downloaded here: 

http://icvl.cs.bgu.ac.il/hyperspectral/

Store this data in a folder called `Images_RAD`. The images will come as *.mat files with arbitrary titles. Renumber them (in order to be fed in the Neural Network) by running ``` Originial_RAD.m ``` on matlab. 

## Requirements 

- Python 3.6 (only tested with 3.6.3)
- Tensorflow >= 1.0 (tested with 1.2.0)
- GPU environment recomended

To install tensorflow, type: 

``` pip install tensorflow==1.2 ```
or
``` pip install tensorflow-gpu==1.2 ```

## Structure
The repository is structured as follows: 

Data*: These folders contain some information about the data (RGB images) that was used in this project. The images haven't been pushed on this repository because it would have exceeded the allowed storage. You won't need this, since you'll have to generate your own data. 

Models: In this folder you will find the necessary code to 
	1) Create your own dataset (python Create_Data*.py)
	2) Build a classifier model to predict the correct CSS parameters from the RGB images (python Main.py).

Preprocessing_Matlab: This folder contains some matlab script which have been used to analyse the hyperspectral images. 

## Getting the code

Clone the repository by typing

``` git clone https://github.com/dariopa/CSS-Reconstruction.git ```

## Running the code locally

1) Open for instance the file `Models/ClassifyBellCurve/Create_Data.py` and edit all paths there to match your system. When you're done, type ``` python Create_Data.py ``` - this will create your RGB dataset. 
2) Next, open `Models/ClassifyBellCurve/Main.py` and edit your path and parameters from line 18 to line 31. When you're done, type ``` python Main.py ``` - this will start your training. Keep in mind to comment line 3 if you're not using GPU environment. 
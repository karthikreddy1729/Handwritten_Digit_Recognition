# Handwritten_Digit_Recognition
The handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a hard task for the machine because handwritten digits are not perfect and can be made with many different flavors. The handwritten digit recognition is the solution to this problem which uses the image of a digit and recognizes the digit present in the image.

*  This model is the implementation of a handwritten digit recognition model using the MNIST dataset.
*  The model is built using Convolutional Neural Networks to identify the digits.
*  In this even the GUI is built in which you can draw the digit and recognize it. After recognizing it displays the recognized digit and the accuracy of that digit on the right side.
*  You can even load any image and recognize the digit in that image using image_recognizer.py file.

## Requirements
*  Python 2.7
*  Tkinter
*  Keras
*  Tensorflow
*  numpy

## Algorithm
*  Convolutional Neural Networks are used in this model to recognize handwritten digits.

* The model is saved as a json file 'model' and the weights are saved in 'mode.h5' and stored in the folder model

## Downloading the dataset
you can download the dataset from the link given (https://www.kaggle.com/c/digit-recognizer/data) ::
* =>Download the dataset and put it in a folder named dataset

The MNIST dataset contains 60,000 training images of handwritten digits from zero to nine and 10,000 images for testing. So, the MNIST dataset has 10 different classes. The handwritten digits images are represented as a 28×28 matrix where each cell contains grayscale pixel value.

The dataset contains 2 csv files:

* train.csv
* test.csv

## Training the data
```
    python mnist_training.py
```

## To test the model using gui
~~~
    python recognizer_gui.py
~~~


## To test the model by loading image
~~~
    python image_recognizer.py
~~~

*  The accuracy of the model is 99.26%  on the test dataset.



# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:14:59 2020

@author: karth
"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import pandas as pd


traindf=pd.read_csv("./dataset/train.csv")
testdf=pd.read_csv("./dataset/test.csv")

Y_train=traindf.iloc[:,0:1].values
X_train=traindf.iloc[:,1:28*28+1].values

y_test=testdf.iloc[:,0:1].values
x_test=testdf.iloc[:,1:28*28+1].values


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
x_test = x_test.astype('float32')
X_train=X_train/255
x_test=x_test/255

input_ = (28,28,1)
n_classes=10
batchsize = 128
epochs = 25

Y_train = keras.utils.to_categorical(Y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


classifier=Sequential()

classifier.add(Conv2D(64,(3,3),activation='relu',input_shape=input_))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(64,(3,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

# classifier.add(Conv2D(64,(3,3),activation='relu'))

# classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dropout(0.3))

classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dropout(0.3))

classifier.add(Dense(units=n_classes,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


classifier.fit(X_train,Y_train,batch_size=batchsize,epochs=epochs,verbose=1,validation_data=(x_test, y_test))



evaluation=classifier.evaluate(x_test,y_test,verbose=0)
print('Test dataset loss:', evaluation[0])
print('Test dataset accuracy:', evaluation[1]*100,"%")


model_json = classifier.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)

print("Save weights to file...")
classifier.save_weights('model.h5',overwrite=True)























# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:14:59 2020

@author: karth
"""

from keras.models import model_from_json
import numpy as np

def loadModel(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_path)
    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = loadModel('./model/model.json', "./model/model.h5")



from PIL import Image 
img = Image.open("./sample_images/zero.png") # open colour image
img = img.resize((28,28))
#convert rgb to grayscale
img = img.convert('L')
img = np.array(img)
#reshaping to support our model input and normalizing
img = img.reshape(1,28,28,1)
img = img/255.0
#predicting the class
res = model.predict([img])[0]

print("The recognized digit is ",np.argmax(res))
print("The accuracy of recognized digit is ",max(res))

















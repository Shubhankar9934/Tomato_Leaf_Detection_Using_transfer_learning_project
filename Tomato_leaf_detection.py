# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 04:01:14 2022

@author: shubhankar kumar
"""

# Tomato Leaf Detection

#Import all dependencies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

path = "C:\\Users\\shubh\\Desktop\\Tomato\\archive (3)\\New Plant Diseases Dataset(Augmented)"
os.listdir(path)

# Join train and test with path
train_path = os.path.join(path, "train")
print(os.listdir(train_path))

test_path = os.path.join(path,"valid")
print(os.listdir(test_path))

# glob Function(The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell)
from glob import glob 
folders = glob("C:\\Users\\shubh\\Desktop\\Tomato\\archive (3)\\New Plant Diseases Dataset(Augmented)\\train\\*")
folders

import matplotlib.pyplot as plt
plt.imshow(plt.imread("C:\\Users\\shubh\\Desktop\\Tomato\\archive (3)\\New Plant Diseases Dataset(Augmented)\\train\\Tomato___Bacterial_spot\\00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp 3110.JPG"))
plt.title("Bacterial spot")

import matplotlib.pyplot as plt
plt.imshow(plt.imread("C:\\Users\\shubh\\Desktop\\Tomato\\archive (3)\\New Plant Diseases Dataset(Augmented)\\train\\Tomato___Early_blight\\0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG"))
plt.title("Early Blight")

import matplotlib.pyplot as plt
plt.imshow(plt.imread("C:\\Users\\shubh\\Desktop\\Tomato\\archive (3)\\New Plant Diseases Dataset(Augmented)\\train\\Tomato___Late_blight\\0003faa8-4b27-4c65-bf42-6d9e352ca1a5___RS_Late.B 4946.JPG"))
plt.title("Late blight")

# import necessary packages

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

# re-size all the images to this 
IMAGE_SIZE = [224,224]

# Import the InceptionV3 Library and add preprocessing layer tp the the from of VGG
# Here we will be using imagenet weights

# InceptionV3 with input shape and weight is imagenet

inception = InceptionV3(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)


# Don't train existing weights 
for layer in inception.layers:
    layer.trainable = False
    
# our layers - I can add more if i want show i am flattening the InceptionV3 
x= Flatten()(inception.output)
# I have multi categories that's why i am using Softmax else using sigmoid.
# Also Concatination with the flatten layer 
prediction = Dense(len(folders),activation = 'softmax')(x)
    
# create a model object 
model = Model(inputs = inception.input,outputs = prediction)

# view the structure 

model.summary()

# Tell The model what cost and optimization Method to use 
model.compile(
    loss = "categorical_crossentropy",
    optimizer = 'adam',
    metrics = ['accuracy']
    )

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True
                                  )
test_datagen = ImageDataGenerator(rescale = 1.0/255)

training_set = train_datagen.flow_from_directory("C:\\Users\\shubh\\Desktop\\Tomato\\archive (3)\\New Plant Diseases Dataset(Augmented)\\train",
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory("C:\\Users\\shubh\\Desktop\\Tomato\\archive (3)\\New Plant Diseases Dataset(Augmented)\\valid",
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

model = model.save("InceptionV3_tomato_leaf_detection.h5")

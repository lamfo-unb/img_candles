# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 11:01:45 2021

@author: vitor
"""

########## RNC
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',
              optimizer = 'Adamax',
              metrics = ['accuracy'])
batch_size = 100
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)
# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('Train/', target_size = (300, 300), 
                                                    batch_size = batch_size,
                                                    class_mode = 'binary') 
# Generator for validation data
validation_generator = test_datagen.flow_from_directory('Test/', 
                                                        target_size = (300, 300),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')
# Fit the model on Training data
model.fit(train_generator,
                    epochs = 5,
                    validation_data = validation_generator,
                    verbose = 1)
# Evaluating model performance on Testing data
loss, accuracy = model.evaluate(validation_generator)
print("\nModel's Evaluation Metrics: ")
print("---------------------------")
print("Accuracy: {} \nLoss: {}".format(accuracy, loss))
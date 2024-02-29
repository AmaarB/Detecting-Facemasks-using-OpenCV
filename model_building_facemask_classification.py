from re import X
from tkinter import Y
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# specify paths to the folders 'with_mask' and 'without_mask'
faceimages_paths = ["with_mask/",
                    "without_mask/"]


images = []
labels = []
# The follwoing 5 lines of code explores the directory faceimages_paths, in particular [0] which is with-mask and appends it to two lists, images and labels
# 0: with mask
for p in os.listdir(faceimages_paths[0]):
  # read image from disk
  img = cv2.imread(faceimages_paths[0]+p)
  # resize image
  img = cv2.resize(img, (20, 20))
  # add loaded image in the list
  images.append(img)
  # label for proper mask
  labels.append(0)

# 1: No Mask
# the follwing code explores the directory faceimages_paths, in particular [1] which is without_mask it resizes appends it to two lists, images and labels
for p in os.listdir(faceimages_paths[1]):
  # read image from disk
  img = cv2.imread(faceimages_paths[1]+p)
  # resize image
  img = cv2.resize(img, (20, 20))         
  # add loaded image in the list
  images.append(img)
  # label for no mask
  labels.append(1)

# for most image data, pixel value are intergers between 0 and 255, large integer values can disrupt the learning process within nueral networks, therfore we
# divide all pixel values by the largest value 255
# scale images to the [0, 1] range
images_scaled = [i.astype("float32")/255 for i in images]

# split data into training and validation sets
# a good test size to use is approximatly 70% for training and 30% for validation
# the random state is used to shuffle the data randomly in the dataset in order to remove biases in the data prediction.
train_images, val_images, y_train, y_val = train_test_split(images_scaled, labels, 
                                                            test_size=0.33, 
                                                            random_state=42)

# convert lists to arrays
train_images = np.array(train_images)
val_images = np.array(val_images)

y_train = np.array(y_train)
y_val = np.array(y_val)

num_classes = 2

# model architecture
model = keras.Sequential(
    [
        keras.Input(shape=(20,20,3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
# Flowing between layers are tensors. Tensors can be seen as shapes. The input layer as shown in the jpg attached
#https://d1zx6djv3kb1v7.cloudfront.net/wp-content/media/2019/05/Hidden-layrs-1-i2tutorials.jpg is the a tensor. This tensor
#must have the same shape as the training data. 
# this code training model has 853 images which where resized to the form of 20 * 20 pixels in RGB(3 chanels) therfore shape = (20,20,3)

#The Keras Conv2d parameter filters and determines the number of kernels to convolve with the input volume
#  layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
# this means 32 filters this number may alter depending on the complexity of the dataset and depth of neural network
# kernel_size is the paramter used to specify the hight and width of the 2d conventional window.

#layers.flatten() converts a matrix into a single array
# layers.dropout(0.5) refers to dropping out hidden and visible units in a neural network and the most common probability is 0.5
#layers.dense applies weight to all nodes from previous layers through matrix vector multiplication

model.summary()

batch_size = 128
# The batch size depicts the number of samples that will be propogated through the netowork
# dataset of 853, algorithm takes first 128 images, trains, then takes images from 129 - 257 and trains
epochs = 20
# epoch means one complete cycle of trainng the neural network

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#loss function - how accuracte the model is during training
              # - sparse_categorical_crossentropy - caluclates the loss between labels and predictions
# Optimizer - how the model updates based of the data it recognizes and its loss function
          # - "Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments." 
          # Quoted from(https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) 
#metrics - is used to monitor the trainng and testing steps, ['accuracy'] is used and it calculates how often predictions equal the labels

# save model after every epoch
callbacks = [keras.callbacks.ModelCheckpoint("model.h5")]

# start model training
history = model.fit(train_images, y_train, batch_size=batch_size, 
          epochs=epochs, validation_data= (val_images, y_val), 
          callbacks=callbacks)
        

X = train_images
Y = y_train

# Graphing Accuracy of Model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Facemask Machine Learning Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.show()

# Graphing Loss of Model

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Facemask Machine Learning Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'])
plt.show()

import os, sys # For handling directories
from random import randint

import numpy as np # We'll be storing our data as numpy arrays
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras.utils import to_categorical
from keras import layers
from keras import models

# Current working directory
sys.path.append(os.getcwd())

lookup = dict() 
reverselookup = dict()
count = 0

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Resources', 'FruQ-multi')

# Assuming BananaDB and CucumberQ are the folders of interest
folders_of_interest = ['BananaDB', 'CucumberQ', 'GrapeQ', 'KakiQ', 'PapayaQ', 'PeachQ', 'PearQ', 'PepperQ', 'StrawberryQ', 'tomatoQ', 'WatermeloQ']

# Initialize lookup dictionaries
for folder in folders_of_interest:
    for category in os.listdir(os.path.join(base_path, folder)):
        if not category.startswith('.'): # Ensure you aren't reading in hidden folders
            lookup[category] = count
            reverselookup[count] = category
            count = count + 1

x_data = []
y_data = []
datacount = 0 # Count the amount of images in the dataset

# Loop over the folders of interest
for folder in folders_of_interest: 
    print("Processing folder: ", folder)

    for category in os.listdir(os.path.join(base_path, folder)):
        if not category.startswith('.'): # Again avoid hidden folders
            print("Processing category: ", category)
            count = 0 # Count how many images are there per gesture type

            category_path = os.path.join(base_path, folder, category)
            for image_name in os.listdir(category_path): 
                print("Processing image: ", image_name)
                
                img_path = os.path.join(category_path, image_name)
                img = Image.open(img_path).convert('L') # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1

                y_values = np.full((count, 1), lookup[category]) 
                y_values = y_values.reshape(-1)
                y_data.append(lookup[category]) 

            datacount = datacount + count

print("Number of Images: ", datacount)
x_data = np.array(x_data, dtype='float32')
y_data = np.array(y_data, dtype='int32')

# Ensure that y_data has the same length as x_data
assert len(x_data) == len(y_data), "x_data and y_data lengths do not match!"

num_classes = len(np.unique(y_data))
print("Number of classes: ", num_classes)

y_data = to_categorical(y_data, num_classes=num_classes)
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255

x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size = 0.2)
x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size = 0.5)

# Check number of GPU's available to train model
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPU's available: ", len(tf.config.list_physical_devices('GPU')))
print("Devices: ", physical_devices)

# Train model using GPU
try:
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
    pass

# Build neural network and classification system
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.add(layers.Dense(num_classes, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Plot Training History
def plot_history(history):
    plt.figure(figsize = (12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Training Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

history = model.fit(x_train, y_train, epochs = 20, batch_size = 64, verbose = 1, validation_data = (x_validate, y_validate))

plot_history(history)

# Save the model for use in other scripts
model.save('Resources/Models/test.h5')
print("Model saved successfully")

# Save dictionaries in pickle files
with open('Resources/Models/lookup.pkl', 'wb') as f:
    pickle.dump(lookup, f)

with open('Resources/Models/reverselookup.pkl', 'wb') as f:
    pickle.dump(reverselookup, f)

[loss, acc] = model.evaluate(x_test, y_test, verbose = 1)
print("Accuracy:" + str(acc))
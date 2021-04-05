import tensorflow as tf

import cv2 as cv
import numpy as np
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
from tensorflow.keras import  layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import IPython.display as display
from model import MyModel as DigitClassificationModel
import dataset
import os
import glob 


BATCH_SIZE = 32
EPOCHS = 5
TARGET_HEIGHT = 96
TARGET_WIDTH = 96
P_LEARNING_RATE = 0.001
P_INITIAL_EPOCH = 0
NUM_CLASSES = 12


FOLDER_TF_RECORDS = "/home/dlar/tensorflow2/applications/Digit_Recognition/digits_tfrecords/"

train_files =  [f for f in glob.glob(FOLDER_TF_RECORDS + 'digits_train*')]
validation_files =  [f for f in glob.glob(FOLDER_TF_RECORDS + 'digits_validation*')]

digits_dataset = dataset.Dataset()
digits_dataset.setParameters(NUM_CLASSES,TARGET_HEIGHT,TARGET_WIDTH)
train_tf_dataset, val_tf_dataset = digits_dataset.setDataset(train_files, validation_files)

# Batch datasets
length_train_dataset = len(list(train_tf_dataset.as_numpy_iterator()))
length_val_dataset = len(list(val_tf_dataset.as_numpy_iterator()))

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_tf_dataset = train_tf_dataset.cache().shuffle(buffer_size=2000, reshuffle_each_iteration=True).prefetch(buffer_size=AUTOTUNE).batch(32)
val_tf_dataset = val_tf_dataset.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Repeating dataset for all epochs
train_tf_dataset = train_tf_dataset.repeat(EPOCHS)
val_tf_dataset = val_tf_dataset.repeat(EPOCHS)

print("\n --------------------------------- \n" 
      "Datasets have been fixed \n "
      "-------------------------\n")

STEP_PER_EPOCH_TRAIN = length_train_dataset/BATCH_SIZE
STEP_PER_EPOCH_VAL = length_val_dataset/BATCH_SIZE

# Create a model
network_input = tf.keras.layers.Input(shape = (96,96,3))
model = DigitClassificationModel()
output_layer = model(network_input)
# Set a compiler
model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
optimizer = tf.keras.optimizers.Adam(P_LEARNING_RATE),
metrics = ['accuracy'])

history = model.fit(
train_tf_dataset,
batch_size = 32,
shuffle=True,
initial_epoch = P_INITIAL_EPOCH,
steps_per_epoch = STEP_PER_EPOCH_TRAIN,
epochs = EPOCHS,
validation_data = val_tf_dataset,
validation_steps = STEP_PER_EPOCH_VAL,
validation_freq=1
)

#plotting loss and accuracy of the model

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
#For smoothing the plots
def smooth_curve(points, factor = 0.6):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

epochs =range(1, len(acc)+1)

plt.plot(epochs, smooth_curve(acc), 'r-', label='Training accuracy')
plt.plot(epochs, smooth_curve(val_acc), 'b-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'r-', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

model.save('desired_digitLENETSlowFeatureExtraction', save_format = 'tf')

import tensorflow as tf 
from tensorflow.keras.applications import MobileNetV2
import numpy as np 


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__() 
        self.base = MobileNetV2(input_shape=(96,96,3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling='same')
        self.base.trainable = False
        self.flattening_layer = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation = tf.nn.relu, use_bias=True, bias_initializer='zeros', kernel_initializer= tf.keras.initializers.GlorotUniform())
        self.dense2 = tf.keras.layers.Dense(12, activation = tf.nn.softmax, use_bias=True, bias_initializer='zeros', kernel_initializer= tf.keras.initializers.GlorotUniform())

    def call(self, inputs):
        convolutional_output = self.base(inputs)
        flattened_layer = self.flattening_layer(convolutional_output)
        x = self.dense1(flattened_layer)
        x = self.dense2(x)           
        return x 



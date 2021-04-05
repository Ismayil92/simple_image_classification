import tensorflow as tf 
import numpy as np

class Dataset:
    def __init__(self):
        self.NUM_CLASSES = 0 
        self.train_tf_dataset = 0
        self.val_tf_dataset = 0
        self.TARGET_HEIGHT = 0
        self.TARGET_WIDTH = 0

    def setParameters(self, p_num_classes, p_target_size_height, p_target_size_width):
        self.NUM_CLASSES = p_num_classes
        self.TARGET_HEIGHT = p_target_size_height 
        self.TARGET_WIDTH = p_target_size_width

    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.cast(image, tf.float32) 
        image = image/255.0
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image,[self.TARGET_HEIGHT,self.TARGET_WIDTH])    
        return image

    def extract_fn(self, data_record):
        features = {
            # Extract features using the keys set during creation
            "image/class/label":    tf.io.FixedLenFeature([], tf.int64),
            "image/encoded":        tf.io.FixedLenFeature([], tf.string),
        }
        sample = tf.io.parse_single_example(data_record, features)
        image = self.decode_image(sample['image/encoded'])
        label = sample['image/class/label'] 
        label = tf.one_hot(label, self.NUM_CLASSES)   
        return image,label

    def setDataset(self, train_files, validation_files):
        self.train_tf_dataset = tf.data.TFRecordDataset(train_files)
        self.val_tf_dataset = tf.data.TFRecordDataset(validation_files)

        self.train_tf_dataset = self.train_tf_dataset.map(self.extract_fn)
        self.val_tf_dataset = self.val_tf_dataset.map(self.extract_fn)

        return self.train_tf_dataset, self.val_tf_dataset

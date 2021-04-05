#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf 
import numpy as np 
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import cv2 as cv
import math

#PATH_TO_CKPT = "/home/dlar/catkin_ws/src/ui_interpretation/Data/object_det_database/wash_detect_2.pb"
PATH_TO_LABELS = '/home/dlar/catkin_ws/src/ui_interpretation/Data/object_det_database/object-detection.pbtxt'
PATH_TO_CKPT = "/home/dlar/catkin_ws/src/ui_interpretation/Data/graph/frozen_mobilenetV2_10_96_gray.pb"



detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)
for op in detection_graph.get_operations():
    print(op.name)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

image_tensor = detection_graph.get_tensor_by_name('MobilenetV2/input:0')
feature_map_tensor = detection_graph.get_tensor_by_name('MobilenetV2/expanded_conv_3/expansion_output:0')
#detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
#detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#detection_scores = detection_graph.get_tensor_by_name('detection_classes:0')
#num_detections = detection_graph.get_tensor_by_name('num_detections:0')


#input image
#img = cv.imread("/home/dlar/models/research/object_detection/data/images/train/frame20325.jpg", cv.IMREAD_COLOR) 
img = cv.imread("/home/dlar/tensorflow2/applications/eight.jpg")
resizedRGBImg = cv.resize(img,(96,96), interpolation=cv.INTER_AREA)
np_image_data = np.asarray(resizedRGBImg)
frame_expanded = np.expand_dims(np_image_data, axis = 0)
#(feature_map, boxes, scores, classes, num) = sess.run([feature_map_tensor, detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: frame_expanded})
feature_map = sess.run(feature_map_tensor, feed_dict={image_tensor: frame_expanded})
print(feature_map.shape)

# print tensor ( without :0 you will get the operation itself )
#print(detection_graph.get_tensor_by_name("{}:0".format(OP_NAME)))
#feature_map = feature_map.squeeze(axis=0)
output = feature_map[0,:,:,:]
#output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
fig = plt.figure(figsize=(24,24))
plt.rcParams['toolbar'] = 'None'

n_columns=10
n_rows = math.ceil(100/n_columns) + 1

for i in range(1, 61):
   # plt.subplot(n_rows, n_columns, i+1)
    fig.add_subplot(n_rows,n_columns, i)
    output_img = cv.normalize(output[:,:,i], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    output_img = cv.resize(output_img,(24,24), interpolation=cv.INTER_NEAREST)
    plt.imshow(output_img, interpolation="nearest", cmap="gray")
plt.show()

plt.savefig('myfigure2.png')


#feature_map = feature_map.squeeze(axis=2)
print(feature_map.shape)
#cv.imshow("picture", output)
cv.waitKey()


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import nibabel as nib
data_dir = 'roiBlocks'
out_dir = 'deepLearn/'
seriesId = '428AB502' # >>> Using first 1200!!! # '95052AB8' # '161066A2'
ctSamples = 300
roiNum = '3'
alt_imageIds = []
num_class = 5
nLabel = 5

width = 64 #256
height = 64 #256
depth = 32 #40
modelId = 'organs-5c-30fs-acc92-121.meta'

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(modelId)
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()

x = graph.get_tensor_by_name("x_start:0")
keepProb = graph.get_tensor_by_name("keepProb:0")
cons1 = graph.get_tensor_by_name("cons1:0")

#Now, access the op that you want to run. 
restoredGetFeats = graph.get_tensor_by_name("getFeats:0")

for op in graph.get_operations():
    if 'Conv3D' in op.type or 'MaxPool3D' in op.type or 'Relu' in op.type:
        print(op.name, op.outputs[0].shape)
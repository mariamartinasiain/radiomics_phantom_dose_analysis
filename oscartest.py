import tensorflow.compat.v1 as tf

from qa4iqi_extraction.constants import MANUFACTURER_FIELD, MANUFACTURER_MODEL_NAME_FIELD, SERIES_DESCRIPTION_FIELD, SERIES_NUMBER_FIELD, SLICE_THICKNESS_FIELD
from swinunetr import custom_collate_fn, load_data
tf.disable_v2_behavior()
import os
import numpy as np
import nibabel as nib
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
from swinunetr import CropOnROId
from monai.data import SmartCacheDataset, DataLoader

def test():
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

def run_inference():
    
    jsonpath = "./dataset_info.json"
    # Define the path to your model files
    model_dir = './'
    model_file = model_dir + 'organs-5c-30fs-acc92-121.meta'

    # Start a new session
    sess = tf.Session()

    # Load the graph
    saver = tf.train.import_meta_graph(model_file)
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    # Access the graph
    graph = tf.get_default_graph()
    feature_tensor = graph.get_tensor_by_name('MaxPool3D_1:0')

    #names for input data and dropout:
    x = graph.get_tensor_by_name('x_start:0') 
    keepProb = graph.get_tensor_by_name('keepProb:0')  


    target_size = (64, 64, 32)
    transforms = Compose([
        LoadImaged(keys=["image", "roi"]),
        EnsureChannelFirstd(keys=["image", "roi"]),
        CropOnROId(keys=["image"], roi_key="roi", size=target_size),
        ToTensord(keys=["image"]),
    ])

    datafiles = load_data(jsonpath)
    dataset = SmartCacheDataset(data=datafiles, transform=transforms, cache_rate=0.05, progress=True, num_init_workers=8, num_replace_workers=8)
    dataload = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=4)


    with open("deepfeaturesoscar.csv", "w", newline="") as csvfile:
        fieldnames = ["SeriesNumber", "deepfeatures", "ROI", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        dataset.start()
        i = 0
        for batch in tqdm(dataload):
            image_tf = tf.transpose(batch["image"], [0, 2, 3, 4, 1])
            
            # Run TensorFlow session to extract features
            features = sess.run(feature_tensor, feed_dict={x: image_tf, keepProb: 1.0})  # Ensure placeholders match
            
            # Process and save features
            latentrep = tf.reshape(features, [-1]).numpy().tolist()
            record = {
                "SeriesNumber": batch["info"][SERIES_NUMBER_FIELD][0],
                "deepfeatures": latentrep,
                "ROI": batch["roi_label"][0],
                "SeriesDescription": batch["info"][SERIES_DESCRIPTION_FIELD][0],
                "ManufacturerModelName": batch["info"][MANUFACTURER_MODEL_NAME_FIELD][0],
                "Manufacturer": batch["info"][MANUFACTURER_FIELD][0],
                "SliceThickness": batch["info"][SLICE_THICKNESS_FIELD][0],
            }
            writer.writerow(record)
           
            if i % 64 == 0:
                dataset.update_cache()
            i += 1
        dataset.shutdown()

    print("Done!")


if __name__ == "__main__":
    #test()
    run_inference()
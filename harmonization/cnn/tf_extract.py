import sys
sys.path.append('/home/reza/radiomics_phantom/')
import tensorflow.compat.v1 as tf

from qa4iqi_extraction.constants import MANUFACTURER_FIELD, MANUFACTURER_MODEL_NAME_FIELD, SERIES_DESCRIPTION_FIELD, SERIES_NUMBER_FIELD, SLICE_THICKNESS_FIELD
from harmonization.swin.extract import custom_collate_fn, load_data
tf.disable_v2_behavior()
import os
import numpy as np
import nibabel as nib
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord,EnsureTyped
from harmonization.swin.extract import CropOnROId
from monai.data import SmartCacheDataset, DataLoader,ThreadDataLoader
import torch
import tensorflow as tf2
gpus = tf2.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf2.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
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
    
    jsonpath = "./dataset_info_cropped.json"
    jsonpath = "./train_configurations/expanded_registered_light_dataset_info.json"
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
    
    
    try:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Calculer le nombre total de poids
        total_params = sum([sess.run(tf.size(var)) for var in variables])
        print(f"Le nombre total de poids dans le modèle est : {total_params}")
    except:
        print("Error while calculating the number of parameters in the model")
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    target_size = (64, 64, 32)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], device=device, track_meta=False),
    ])

    datafiles = load_data(jsonpath)
    dataset = SmartCacheDataset(data=datafiles, transform=transforms,cache_rate=1,progress=True,num_init_workers=8, num_replace_workers=8,replace_rate=0.1)
    print("dataset length: ", len(datafiles))
    dataload = ThreadDataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    with open("normalized_deepfeaturesoscar.csv", "w", newline="") as csvfile:
        fieldnames = ["SeriesNumber", "deepfeatures", "ROI", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        dataset.start()
        input_tensor = tf.placeholder(tf.float32, shape=[None, 1, 64, 64, 32])
        image_tf = tf.transpose(input_tensor, [0, 2, 3, 4, 1])  # Reorder dimensions
        i = 0
        #iterating over every entry in the data
        iterator = iter(dataload)
        for _ in tqdm(range(len(datafiles))):
            batch = next(iterator)                      
            flattened_image = tf.reshape(image_tf, [-1, 131072])  # Flatten the tensor to match the input placeholder
            # Extract features using the reshaped tensor
            features = sess.run(feature_tensor, feed_dict={x: sess.run(flattened_image, feed_dict={input_tensor: batch["image"].numpy()}), keepProb: 1.0})
            
            # Process and save features
            latentrep = sess.run(tf.reshape(features, [-1])).tolist()
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
           
            
        dataset.shutdown()

    print("Done!")


if __name__ == "__main__":
    #test()
    run_inference()


'''

import sys
sys.path.append('/home/maria/radiomics_phantom_copy/')
import tensorflow.compat.v1 as tf

from qa4iqi_extraction.constants import MANUFACTURER_FIELD, MANUFACTURER_MODEL_NAME_FIELD, SERIES_DESCRIPTION_FIELD, SERIES_NUMBER_FIELD, SLICE_THICKNESS_FIELD
from harmonization.swin.extract import custom_collate_fn, load_data, CropOnROId
tf.disable_v2_behavior()
import os
import numpy as np
import nibabel as nib
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord,EnsureTyped
from harmonization.swin.extract import CropOnROId
from monai.data import SmartCacheDataset, DataLoader,ThreadDataLoader
import torch
import tensorflow as tf2
gpus = tf2.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf2.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

centersrois = {'cyst1':  [324, 334, 158],'cyst2' :  [189, 278, 185],'hemangioma' :  [209, 315, 159],'metastasis' : [111, 271, 140],'normal1' : [161, 287, 149],'normal2' :  [154, 229, 169]}

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

    jsonpath = "/mnt/nas7/data/maria/final_features/expanded_registered_light_dataset_info.json"
    model_dir = '/mnt/nas7/data/maria/final_features/QA4IQI/QA4IQI/'
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
    
    
    try:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Calculer le nombre total de poids
        total_params = sum([sess.run(tf.size(var)) for var in variables])
        print(f"Le nombre total de poids dans le modèle est : {total_params}")
    except:
        print("Error while calculating the number of parameters in the model")
    device_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    target_size = (64, 64, 32)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], device=device, track_meta=False),
        CropOnROId(keys=["image"],roi_key="roi_label",size=target_size,precomputed=True,centers=centersrois)
    ])

    from collections import defaultdict

    datafiles = load_data(jsonpath)

    # Group by SeriesNumber
    datafiles.sort(key=lambda x: x["info"]["SeriesNumber"])

    # Now use the sorted list
    dataset = SmartCacheDataset(
        data=datafiles, transform=transforms,
        cache_rate=0.1, progress=True,
        num_init_workers=8, num_replace_workers=8, replace_rate=0.1
    )

    print("Dataset length:", len(datafiles))
    dataload = ThreadDataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    output_dir = "/mnt/nas7/data/maria/final_features/"
    output_file = os.path.join(output_dir, "cnn_features_prueba2.csv")

    # Track processed images
    processed_series = set()

    # Open CSV file once
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["SeriesNumber", "deepfeatures", "ROI", "SeriesDescription", "ManufacturerModelName", "Manufacturer", "SliceThickness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        dataset.start()  # Initialize cache

        input_tensor = tf.placeholder(tf.float32, shape=[None, 1, 64, 64, 32])
        image_tf = tf.transpose(input_tensor, [0, 2, 3, 4, 1])  # Reorder dimensions

        while len(processed_series) < len(datafiles):  # Continue until all images are processed
            iterator = iter(dataload)
            for _ in tqdm(range(len(datafiles))):
                try:
                    batch = next(iterator)
                    print("Batch loaded successfully!")
                    
                    series_number = batch["info"]["SeriesNumber"][0]
                    roi = batch["roi_label"][0]  # Extract the ROI label
                    print(f"Processing {series_number}, ROI: {roi}")

                    # Skip if this (SeriesNumber, ROI) pair was already processed
                    if (series_number, roi) in processed_series:
                        continue

                    flattened_image = tf.reshape(image_tf, [-1, 131072])  # Flatten tensor
                    image_cpu = batch["image"].cpu().numpy()  # Move to CPU
                    features = sess.run(feature_tensor, feed_dict={x: sess.run(flattened_image, feed_dict={input_tensor: image_cpu}), keepProb: 1.0})
                    
                    # Process and save features
                    latentrep = sess.run(tf.reshape(features, [-1])).tolist()
                    writer.writerow({
                        "SeriesNumber": batch["info"][SERIES_NUMBER_FIELD][0],
                        "deepfeatures": latentrep,
                        "ROI": batch["roi_label"][0],
                        "SeriesDescription": batch["info"][SERIES_DESCRIPTION_FIELD][0],
                        "ManufacturerModelName": batch["info"][MANUFACTURER_MODEL_NAME_FIELD][0],
                        "Manufacturer": batch["info"][MANUFACTURER_FIELD][0],
                        "SliceThickness": batch["info"][SLICE_THICKNESS_FIELD][0],
                    })

                    processed_series.add((series_number, roi))

                except StopIteration:
                    print("Iterator exhausted, updating cache...")
                    break  # Exit loop and refresh cache

            dataset.update_cache()  # Refresh dataset to load new samples

            #writer.writerow(record)
                 
        dataset.shutdown()

    print("Done!")


if __name__ == "__main__":
    #test()
    run_inference()

'''
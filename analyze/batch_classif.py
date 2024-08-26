from analyze.classification import train_mlp_svm
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUS",gpus)
if gpus:
    try:
        # Set memory growth to avoid taking all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

#configurations is a dictionary with the name of the feature set as key and the latent size as value (used to build the mlp)
configurations = {
    'random_contrast_5_6_lowLR_12batch_swin' : (3072),
    'random_contrast_8_8_swin': (3072),
    'random_contrast_registered' : (3072),


    
}
#classif types can be 'roi_small' for 4 classes , 'roi_large' for 6 classes rois classification , 'scanner' for scanner classification
classif_types = ['roi_small']
mg_filters = [None] #can be used to filter out features with mg values differents form the ones specified in the list if using a dataset of features with different mg values.
for classif_type in classif_types:
    for mg_filter in mg_filters:
        for model, info_list in configurations.items():
            latent_size = info_list
            data_path = f'features_{model}.csv' #the features set must have a name that matches the key in the configurations dictionary and this pattern
            mlp_accuracy,mlp_max_accu,mlp_min_accu = train_mlp_svm(latent_size, data_path, "",classif_type,mg_filter=mg_filter)
            #_,svm_accuracy,svm_max_accu,svm_min_accu = train_svm(data_path,classif_type,mg_filter=None)

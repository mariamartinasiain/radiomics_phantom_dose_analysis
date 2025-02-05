import os
import sys
sys.path.append('/home/reza/radiomics_phantom/')
sys.path.append('/home/reza/radiomics_phantom/analyze')

from classification import train_mlp_svm
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
    'random_contrast_registered' : (3072)    
}
configurations = {
    './train_configurations/liverrandom_contrast_5_15_10batch_swin' : (3072)    
}
configurations = {
    './train_configurations/pyradiomics_full' : (87),
    './train_configurations/oscar_full' : (2048),
    './train_configurations/swinunetr_full' : (3072),
    # './train_configurations/swinunetr_contrastive_full' : (3072),
    # './train_configurations/swinunetr_contrastive_full_loso' : (3072)  
}

#results_dir = '/home/reza/radiomics_phantom/results'
results_dir = '/home/reza/radiomics_phantom/final_features/small_roi'
# results_dir = '/home/reza/radiomics_phantom/final_features/small_roi_combat'
os.makedirs(results_dir, exist_ok=True) 

#classif types can be 'roi_small' for 4 classes , 'roi_large' for 6 classes rois classification , 'scanner' for scanner classification
classif_types = ['roi_small']#['scanner']#
mg_filters = [10] #can be used to filter out features with mg values differents form the ones specified in the list if using a dataset of features with different mg values.
for classif_type in classif_types:
    for mg_filter in mg_filters:
        for model, info_list in configurations.items():
            latent_size = info_list
            if 'loso' in model:
                if 'combat' not in results_dir:
                    data_path = f'{results_dir}/features_loso/features_liverrandom_contrast_5_15_10_batch_swin_loso_00.csv' #the features set must have a name that matches the key in the configurations dictionary and this pattern
                else:
                    data_path = f'{results_dir}/features_loso/features_liverrandom_contrast_5_15_10_batch_swin_loso_00_2combat.csv' #the features set must have a name that matches the key in the configurations dictionary and this pattern
            else:
                data_path = f'{results_dir}/features_{os.path.basename(model)}.csv'
            mlp_accuracy,mlp_max_accu,mlp_min_accu = train_mlp_svm(latent_size, data_path, "",classif_type,mg_filter=mg_filter)
            #_,svm_accuracy,svm_max_accu,svm_min_accu = train_svm(data_path,classif_type,mg_filter=None)

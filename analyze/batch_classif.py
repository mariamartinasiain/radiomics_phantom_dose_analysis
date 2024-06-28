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

configurations = {
    'pyradiomics': (86),
    'swinunetr_full_averaged' : (768), 
    'oscar_full' : (2048),
}
classif_types = ['scanner']
mg_filters = [None]
for classif_type in classif_types:
    for mg_filter in mg_filters:
        for model, info_list in configurations.items():
            latent_size = info_list
            data_path = f'features_{model}.csv'
            #output_path_mlp = f'classif_models/classifier_{model}_{n_scanners}_{classif_type}_{qmg}_mlp.h5'
            mlp_accuracy,mlp_max_accu,mlp_min_accu = train_mlp_svm(latent_size, data_path, "",classif_type,mg_filter=mg_filter)
            #_,svm_accuracy,svm_max_accu,svm_min_accu = train_svm(data_path,classif_type,mg_filter=None)
            #update_performance_file(model, n_scanners, mlp_accuracy, svm_accuracy, output_path_mlp,classif_type,mlp_max_accu,mlp_min_accu,svm_max_accu,svm_min_accu)

from mlp_classif import train_mlp
from mlp_classif import train_mlp_with_data
from mlp_classif import load_data
from svm_classif import train_svm
from svm_classif import train_svm_with_data

def update_performance_file(model, scanners, mlp_accuracy, svm_accuracy, output_path,classif_type='roi_small'):
    entry = f'''
    {model}_classif_{classif_type}_{scanners}_scanners:
        mlp: {mlp_accuracy:.4f}
        svm: {svm_accuracy:.4f}
        path : {output_path}
    '''
    with open('performance_classif.txt', 'a') as file:
        file.write(entry)

configurations = {
    'pyradiomics': ([1, 3, 5, 8], 86),
    'swinunetr': ([1, 3, 5, 8], 3072)
}
classif_type = 'roi_small'
for model, info_list in configurations.items():
    scanners,latent_size = info_list
    for n_scanners in scanners:
        print(f'Training {model} with {n_scanners} scanners on {classif_type} labels')
        test_size = 1 - (n_scanners/13) - 0.02
        data_path = f'features_{model}.csv'
        output_path_mlp = f'classif_models/classifier_{model}_{n_scanners}scanners_mlp.h5'

        mlp_accuracy = train_mlp(latent_size,test_size, data_path, output_path_mlp,classif_type)
        _,svm_accuracy = train_svm(data_path,test_size,classif_type)

        update_performance_file(model, n_scanners, mlp_accuracy, svm_accuracy, output_path_mlp,classif_type)

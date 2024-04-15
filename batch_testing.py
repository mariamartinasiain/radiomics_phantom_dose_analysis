from mlp_classif import train_mlp
from svm_classif import train_svm

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
    print(info_list)
    for (n_scanners,latent_size) in info_list:
        test_size = 1 - (n_scanners/13)
        data_path = f'data/output/features_{model}.csv'
        output_path_mlp = f'classif_models/classifier_{model}_{n_scanners}scanners_mlp.h5'

        mlp_accuracy = train_mlp(latent_size,test_size, data_path, output_path_mlp,classif_type)
        _,svm_accuracy = train_svm(test_size,data_path,classif_type)

        update_performance_file(model, n_scanners, mlp_accuracy, svm_accuracy, output_path_mlp,classif_type)

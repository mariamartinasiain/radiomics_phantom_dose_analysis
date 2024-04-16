import numpy as np
from sklearn import svm
from mlp_classif import load_data


def train_svm(data_path,test_size,classif_type='roi_small',mg_filter=None):
    features, labels, groups, splits, cw, classes_size = load_data(data_path,test_size,label_type=classif_type,mg_filter=mg_filter,one_hot=False)
    
    mean_val_accuracy = 0
    nsplits = splits.get_n_splits()
    splits = splits.split(features, labels, groups=groups)
    print(f'Going to start training with {nsplits} splits')
    for train_idx, val_idx in splits:
        x_train, x_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        train_groups = np.unique(groups[train_idx])
        val_groups = np.unique(groups[val_idx])
        
        print(f'Loaded {len(x_train)} training samples and {len(x_val)} validation samples')
        print(f'Training groups: {train_groups}')
        print(f'Validation groups: {val_groups}')
        
        clf = svm.LinearSVC()
        clf.fit(x_train, y_train)
        
        accuracy = clf.score(x_val, y_val)
        print(f"Accuracy sur l'ensemble de validation: {accuracy * 100:.2f}%")
        mean_val_accuracy += accuracy

    mean_val_accuracy /= nsplits
    return clf, mean_val_accuracy

def train_svm_with_data(x_train, y_train, x_val, y_val):
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    
    accuracy = clf.score(x_val, y_val)
    print(f"Accuracy sur l'ensemble de validation: {accuracy * 100:.2f}%")

    return clf, accuracy

def main():
    train_svm('data/output/features.csv')

if __name__ == '__main__':
    main()
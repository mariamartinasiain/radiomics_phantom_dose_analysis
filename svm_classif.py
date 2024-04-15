from sklearn import svm
from mlp_classif import load_data


def train_svm(data_path,test_size,classif_type='roi_small'):
    x_train, y_train, x_val, y_val,cw = load_data(data_path,test_size,one_hot=False,label_type=classif_type)
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    
    accuracy = clf.score(x_val, y_val)
    print(f"Accuracy sur l'ensemble de validation: {accuracy * 100:.2f}%")

    return clf, accuracy

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
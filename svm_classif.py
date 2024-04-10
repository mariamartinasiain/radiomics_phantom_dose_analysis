from sklearn import svm
from mlp_classif import load_data


def train_classifier(data_path):
    x_train, y_train, x_val, y_val,cw = load_data(data_path,one_hot=False)
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    
    accuracy = clf.score(x_val, y_val)
    print(f"Accuracy sur l'ensemble de validation: {accuracy * 100:.2f}%")

    return clf, accuracy

def main():
    train_classifier('deepfeatures.csv')

if __name__ == '__main__':
    main()
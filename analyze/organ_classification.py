import tensorflow as tf
from keras import layers
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
from collections import Counter
from sklearn.metrics import classification_report
import json
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(filepath, one_hot=True):
    print(f'Loading data from {filepath}')
    data = pd.read_csv(filepath)
    print(f'Initial length of data: {len(data)}')

    # Combine left/right organs
    data['ROI'] = data['ROI'].replace({
        'kidneys_left': 'kidneys',
        'kidneys_right': 'kidneys',
        'lungs_left': 'lungs',
        'lungs_right': 'lungs'
    })

    # Exclude brain samples
    data = data[data['ROI'] != 'brain']
    print(f'Filtered length (excluding brain): {len(data)}')

    labels = data['ROI'].values

    features = data.drop(columns=['FileName', 'ROI'], errors='ignore')

    if 'deepfeatures' in data.columns:
        features = data['deepfeatures'].apply(eval).apply(pd.Series)
    features = features.values

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print('Number of features:', features.shape[1])

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    if one_hot:
        labels_categorical = to_categorical(labels_encoded)
    else:
        labels_categorical = labels_encoded

    print(f'Labeled classes (after brain exclusion): {label_encoder.classes_}')
    classes_size = len(label_encoder.classes_)

    return data, features, labels_categorical, labels_encoded, classes_size, label_encoder


def define_classifier(input_size, classes_size, classifier_type='mlp'):

    if classifier_type == 'mlp':
        def mlp(x, dropout_rate, hidden_units):
            for units in hidden_units:
                x = layers.Dense(units, activation=tf.nn.gelu)(x)
                x = layers.Dropout(dropout_rate)(x)
            return x

        input = tf.keras.Input(shape=(input_size,))
        ff = mlp(input, 0.2, [100, 60, 30])
        classif = layers.Dense(classes_size, activation='softmax')(ff)

        classifier = tf.keras.Model(inputs=input, outputs=classif)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    elif classifier_type == 'logistic_regression':
        # Logistic Regression classifier
        classifier = LogisticRegression(max_iter=1000)

    elif classifier_type == 'knn':
        # k-NN classifier
        classifier = KNeighborsClassifier(n_neighbors=5)

    elif classifier_type == 'random_forest':
        # Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    return classifier

def check_class_distribution(X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    for train_index, test_index in skf.split(X, y):
        y_train, y_test = y[train_index], y[test_index]
        print(f"Fold {fold}")
        print("Train class distribution:", dict(Counter(y_train)))
        print("Test class distribution: ", dict(Counter(y_test)))
        print("-" * 40)
        fold += 1


def run_cross_validation(filepath, output_dir, classifier_type='mlp', n_splits=10):
    os.makedirs(output_dir, exist_ok=True)

    data, features, labels_cat, labels_int, classes_size, label_encoder = load_data(filepath, one_hot=True)
    input_size = features.shape[1]

    # Check if class distribution is balanced across folds
    check_class_distribution(features, labels_int, n_splits=n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    all_fold_precisions = []
    all_fold_recalls = []
    all_fold_f1s = []
    classification_reports_per_fold = {}

    for fold_idx, (train_index, test_index) in enumerate(skf.split(features, labels_int), 1):
        print(f"\n=== Fold {fold_idx}/{n_splits} ===")
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels_cat[train_index], labels_cat[test_index]

        if classifier_type == 'mlp':
            classifier = define_classifier(input_size, classes_size, classifier_type='mlp')
            history = classifier.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                batch_size=8,
                epochs=30,
                verbose=1,
            )
            y_pred = classifier.predict(X_test)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_test, axis=1)
        else:
            classifier = define_classifier(input_size, classes_size, classifier_type)
            classifier.fit(X_train, np.argmax(y_train, axis=1))
            y_pred_labels = classifier.predict(X_test)
            y_true_labels = np.argmax(y_test, axis=1)


        #acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
        acc = np.mean(y_pred_labels == y_true_labels)
        print(f"Accuracy for Fold {fold_idx}: {acc:.4f}")
        fold_accuracies.append(acc)

        report_dict = classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True
        )

        # Store full classification report dict as JSON for this fold
        classification_reports_per_fold[f'fold_{fold_idx}'] = report_dict

        # Collect macro avg metrics for summary
        fold_precision = report_dict['macro avg']['precision']
        fold_recall = report_dict['macro avg']['recall']
        fold_f1 = report_dict['macro avg']['f1-score']

        all_fold_precisions.append(fold_precision)
        all_fold_recalls.append(fold_recall)
        all_fold_f1s.append(fold_f1)

    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    avg_precision = np.mean(all_fold_precisions)
    std_precision = np.std(all_fold_precisions)

    avg_recall = np.mean(all_fold_recalls)
    std_recall = np.std(all_fold_recalls)

    avg_f1 = np.mean(all_fold_f1s)
    std_f1 = np.std(all_fold_f1s)

    print(f"\n=== Final Results ===")
    print(f"Mean Accuracy: {avg_accuracy:.4f}")
    print(f"Std Dev: {std_accuracy:.4f}")

    # Save CSV summary file
    method_name = os.path.basename(filepath).split('.')[0]
    result_path = os.path.join(output_dir, f'results_{method_name}_{classifier_type}.csv')

    results_df = pd.DataFrame({
        'fold': list(range(1, n_splits + 1)) + ['mean', 'std'],
        'accuracy': fold_accuracies + [avg_accuracy, std_accuracy],
        'precision': all_fold_precisions + [avg_precision, std_precision],
        'recall': all_fold_recalls + [avg_recall, std_recall],
        'f1_score': all_fold_f1s + [avg_f1, std_f1]
    })

    results_df.to_csv(result_path, index=False)
    print(f"Summary CSV saved to: {result_path}")

    # Save full classification reports per fold as a separate JSON file
    json_path = os.path.join(output_dir, f'classification_reports_{method_name}_{classifier_type}.json')
    with open(json_path, 'w') as f:
        json.dump(classification_reports_per_fold, f, indent=4)
    print(f"Detailed classification reports saved to: {json_path}")


def main():
    output_dir = '/mnt/nas7/data/maria/final_features/CT-ORG/organ_classification'
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        '/mnt/nas7/data/maria/final_features/pyradiomics_extraction/ct-org/features_pyradiomics_ct-org.csv',
        '/mnt/nas7/data/maria/final_features/cnn/ct-org/features_cnn_org.csv',
        '/mnt/nas7/data/maria/final_features/swinunetr/ct-org/features_swinunetr_org.csv',
        '/mnt/nas7/data/maria/final_features/ct-fm/ct-org/features_ct-fm_org.csv'  
    ]

    classifier_types = ['mlp', 'logistic_regression', 'knn', 'random_forest']

    for path in csv_paths:
        for clf_type in classifier_types:
            run_cross_validation(path, output_dir, classifier_type=clf_type, n_splits=10)

if __name__ == '__main__':
    main()

import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import seaborn as sns
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_data(filepath, one_hot=True):
    print(f'Loading data from {filepath}')
    data = pd.read_csv(filepath)
    print(f'Length of data: {len(data)}')

    # Extract scanner from SeriesDescription
    data['Scanners'] = data['SeriesDescription'].str[:2].values
    print('Scanners: ', np.unique(data['Scanners']))

    # Extract numerical doses from SeriesDescription
    data['Dose'] = data['SeriesDescription'].apply(extract_mg_value)

    # Create the ROI-Scanner pair column
    data['ROI_Scanner_Pair'] = data['Scanners'] + "_" + data['ROI']
    print('Unique ROI-Scanner pairs:', np.unique(data['ROI_Scanner_Pair']))
    print('Number of ROI-Scanner pairs:', len(np.unique(data['ROI_Scanner_Pair'])))

    labels = data['Dose'].values

    # Drop non-relevant columns to extract features
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription',
                                  'ROI', 'ManufacturerModelName', 'Manufacturer',
                                  'SliceThickness', 'SpacingBetweenSlices', 'Scanners', 
                                  'ROI_Scanner_Pair', 'Dose','FileName'], errors='ignore')

    if 'deepfeatures' in data.columns:
        features = features['deepfeatures'].apply(eval).apply(pd.Series)

    features = features.values

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print('Number of features', len(features))

    # Encode labels

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    if one_hot:
        labels = to_categorical(labels)
        
    print(f'Labeled classes: {label_encoder.classes_}')
    classes_size = len(label_encoder.classes_)

    return data, features, labels, classes_size, label_encoder


def extract_mg_value(series_description):
    """Extracts dose (mg value) from the SeriesDescription column if followed by 'mGy'."""
    import re
    match = re.search(r'(\d+)mGy', series_description)
    return int(match.group(1)) if match else None


def define_classifier(input_size, classes_size):
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
    
    return classifier


def train_and_test(X_train, y_train, X_test, y_test, input_size, classes_size, classifier_type='mlp'):

    if classifier_type == 'mlp':
        classifier = define_classifier(input_size, classes_size)
        
        # Fit the model
        history = classifier.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=8,
            epochs=30,
            verbose=1,
        )

        # Predict and calculate accuracy
        y_pred = classifier.predict(X_test)
        val_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) 
        y_true = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        # Plot accuracy and loss
        plot_accuracy_loss(history)

    elif classifier_type == 'logistic_regression':
        # Logistic Regression classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, np.argmax(y_train, axis=1))  # Convert one-hot labels to class indices
        y_pred = clf.predict(X_test)
        val_accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
        y_true = np.argmax(y_test, axis=1)


    elif classifier_type == 'knn':
        # k-NN classifier
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, np.argmax(y_train, axis=1))  # Convert one-hot labels to class indices
        y_pred = clf.predict(X_test)
        val_accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
        y_true = np.argmax(y_test, axis=1)
        
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    return val_accuracy, y_true, y_pred


def plot_confusion_matrix(y_true_all, y_pred_all, class_names, output_dir, method_name, classifier_type):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(10, 8))
    
    # Set color bar limits with vmin and vmax
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1500)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix ({method_name} - {classifier_type})")
    
    # Save the figure
    cm_path = os.path.join(output_dir, f'confusion_matrix_{method_name}_{classifier_type}.png')
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved at: {cm_path}")
    plt.show()


def run_cross_validation(data_path, classifier_type='mlp', output_dir='./results'):
    data, features, labels, classes_size, label_encoder = load_data(data_path, one_hot=True)
    method_name = os.path.basename(data_path).split('_')[1]  # Extract method name

    groups = data['ROI_Scanner_Pair']  # Group by ROI-Scanner pairs
    group_kfold = GroupKFold(n_splits=10)


    fold_accuracies = []
    y_true_all = []
    y_pred_all = []

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'cross_validation_{method_name}_{classifier_type}.txt')

    with open(results_file, 'w') as file:
        file.write('Fold Accuracy Results:\n')

        for fold, (train_idx, test_idx) in enumerate(group_kfold.split(features, labels, groups)):
            print(f"\n=== Fold {fold + 1} ===")

            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            val_accuracy, y_true, y_pred = train_and_test(
                X_train, y_train, X_test, y_test, X_train.shape[1], classes_size, classifier_type
            )

            fold_accuracies.append(val_accuracy)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            print(f"Fold {fold + 1} Accuracy: {val_accuracy:.4f}")
            file.write(f"Fold {fold + 1} Accuracy: {val_accuracy:.4f}\n")

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)

        print(f"\nFinal 10-Fold Cross-Validation Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        file.write(f"\nFinal 10-Fold Cross-Validation Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")

                # Save final result
        file.write(f"\nFinal 10-Fold Cross-Validation Accuracy: {mean_accuracy:.4f}\n")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(1, 11), fold_accuracies, color='skyblue', label="Accuracy per Fold")
        plt.axhline(mean_accuracy, color='r', linestyle='--', label=f"Mean Accuracy: {mean_accuracy:.4f}")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title(f"10-Fold Cross-Validation Accuracy {method_name}")
        plt.legend()
        
        # Add numbers above the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
        
        # Set y-axis range between 0 and 1
        plt.ylim(0, 1)
        
        # Save the plot to a file
        plot_path = os.path.join(output_dir, f'cross_validation_{method_name}_{classifier_type}.png')
        plt.savefig(plot_path)
        print(f"Plot saved at: {plot_path}")
        
        plt.show()

        # Plot and save confusion matrix
        plot_confusion_matrix(y_true_all, y_pred_all, label_encoder.classes_, output_dir, method_name, classifier_type)


def plot_accuracy_loss(history):
    """Plots training and validation accuracy and loss."""
    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.show()



def main():
    files_dir = '/mnt/nas7/data/maria/final_features'
    #files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    output_dir = '/mnt/nas7/data/maria/final_features/dose_classification/six_rois'
    #output_dir = '/mnt/nas7/data/maria/final_features/dose_classification/four_rois'
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        #f'{files_dir}/features_pyradiomics_full.csv',
        #f'{files_dir}/features_cnn_full.csv',
        #f'{files_dir}/features_swinunetr_full.csv',
        f'{files_dir}/features_ct-fm_full.csv'
    ]

    for path in csv_paths:
        run_cross_validation(path, classifier_type='mlp', output_dir=output_dir) 


if __name__ == '__main__':
    main()

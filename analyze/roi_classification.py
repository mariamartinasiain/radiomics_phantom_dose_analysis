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


def load_data(filepath, one_hot=True):
    print(f'Loading data from {filepath}')
    data = pd.read_csv(filepath)
    print(f'Length of data: {len(data)}')
    
    labels = data['ROI'].values

    # Extract numerical doses
    data['Dose'] = data['SeriesDescription'].apply(extract_mg_value)
    print('Unique doses: ', np.unique(data['Dose']))

    # Drop non-relevant columns to extract features
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription',
                                  'ROI','ManufacturerModelName','Manufacturer',
                                  'SliceThickness','SpacingBetweenSlices'],errors='ignore')
    if 'deepfeatures' in data.columns:
        features = features['deepfeatures'].apply(eval).apply(pd.Series)
    features = features.values

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print('Number of features', len(features))

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    if one_hot:
        labels = to_categorical(labels)
    print(f'Labeled classes: {label_encoder.classes_}')
    classes_size = len(label_encoder.classes_)

    return data, features, labels, classes_size

def extract_mg_value(series_description):
    """Extracts dose (mg value) from the SeriesDescription column if followed by 'mGy'."""
    import re
    match = re.search(r'(\d+)mGy', series_description)
    if match:
        return int(match.group(1))
    else:
        return None
    
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
    
    #classifier.summary()
    #print(classifier.summary())
    
    return classifier

# Train and test a classifier
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

    elif classifier_type == 'logistic_regression':
        # Logistic Regression classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, np.argmax(y_train, axis=1))  # Convert one-hot labels to class indices
        y_pred = clf.predict(X_test)
        val_accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))

    elif classifier_type == 'knn':
        # k-NN classifier
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, np.argmax(y_train, axis=1))  # Convert one-hot labels to class indices
        y_pred = clf.predict(X_test)
        val_accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
    
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    return val_accuracy

def run_cross_validation(data_path, output_dir, classifier_type='mlp'):
    start_time_total = time.time()  # Tiempo de inicio global

    # Extract the method name from the file path
    method_name = os.path.basename(data_path).split('_')[1]  # Extracts 'pyradiomics', 'cnn', or 'swinunetr'
    
    data, features, labels, classes_size = load_data(data_path, one_hot=True)
    
    doses = np.unique(data['Dose'])
    results_matrix = np.zeros((len(doses), len(doses)))  # 5x5 matrix
    detailed_results = []  # To store detailed results for each training and testing dose pair
    
    for train_dose_idx, train_dose in enumerate(doses):
        print(f"Training with Dose: {train_dose}")
        start_train_time = time.time()  # Tiempo de inicio de entrenamiento
        
        # Train on the current dose
        X_train = features[data['Dose'] == train_dose]
        y_train = labels[data['Dose'] == train_dose]

        for test_dose_idx, test_dose in enumerate(doses):
            
            print(f"Testing with Dose: {test_dose}")
            start_test_time = time.time()  # Tiempo de inicio de prueba
            
            # Test on the different dose
            X_test = features[data['Dose'] == test_dose]
            y_test = labels[data['Dose'] == test_dose]
            
            print(f"Train Dose: {train_dose} → Samples: {len(X_train)}")
            print(f"Test Dose: {test_dose} → Samples: {len(X_test)}")

            print(f"Train doses: {np.unique(data.loc[data['Dose'] == train_dose, 'Dose'])}")
            print(f"Test doses: {np.unique(data.loc[data['Dose'] == test_dose, 'Dose'])}")

            val_accuracy = train_and_test(X_train, y_train, X_test, y_test, X_train.shape[1], classes_size, classifier_type)
            results_matrix[train_dose_idx, test_dose_idx] = val_accuracy

            test_duration = time.time() - start_test_time  # Tiempo de prueba
            print(f"Results for Train Dose {train_dose} → Test Dose {test_dose}: Accuracy: {val_accuracy:.4f} (Test time: {test_duration:.2f}s)")


            # Save detailed results
            detailed_results.append({
                'train_dose': train_dose,
                'test_dose': test_dose,
                'validation_accuracy': val_accuracy
            })

        train_duration = time.time() - start_train_time  # Tiempo total de entrenamiento
        print(f"Training time for Dose {train_dose}: {train_duration:.2f}s")
    
    # Save detailed results to a CSV file
    detailed_results_df = pd.DataFrame(detailed_results)
    detailed_results_path = os.path.join(output_dir, f'detailed_results_{method_name}_{classifier_type}.csv')
    detailed_results_df.to_csv(detailed_results_path, index=False)
    print(f'Detailed results saved to: {detailed_results_path}')

    # Plot the results as a heatmap
    plot_heatmap(results_matrix, doses, output_dir, method_name, classifier_type)

    total_duration = time.time() - start_time_total
    print(f'Total execution time for {method_name}: {total_duration:.2f}s')


def plot_heatmap(matrix, doses, output_dir, method_name, classifier_type='mlp'):
    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap="Blues", xticklabels=doses, yticklabels=doses, 
                cbar=True, fmt=".3f", vmin=0.70, vmax=1)  # Fix color bar from 0.70 to 1
    plt.title(f"Accuracy Matrix for Doses (Training vs Testing) - {method_name}")
    plt.xlabel("Training Dose (mGy)")
    plt.ylabel("Testing Dose (mGy)")

    # Save the plot
    plot_path = os.path.join(output_dir, f'accuracy_heatmap_{method_name}_{classifier_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # Save as high-quality image
    plt.close()  # Close the plot to free memory
    print(f'Heatmap saved to: {plot_path}')


def main():
    #files_dir = '/mnt/nas7/data/maria/final_features'
    files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    #output_dir = '/mnt/nas7/data/maria/final_features/roi_classification'
    output_dir = '/mnt/nas7/data/maria/final_features/roi_classification/four_rois'
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        #f'{files_dir}/features_pyradiomics_full.csv',
        #f'{files_dir}/features_cnn_full.csv',
        f'{files_dir}/features_swinunetr_full.csv',
        #f'{files_dir}/features_ct-fm_full.csv'
    ]

    for path in csv_paths:
        run_cross_validation(path, output_dir, classifier_type='mlp')
    
if __name__ == '__main__':
    main()

import tensorflow as tf
from keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping


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
                                  'ROI_Scanner_Pair', 'Dose'], errors='ignore')


    if 'deepfeatures' in data.columns:
        features = features['deepfeatures'].apply(eval).apply(pd.Series)

    features = features.values

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    print('Number of features:', features.shape[1])

    return data, features, labels


def extract_mg_value(series_description):
    """Extracts dose (mg value) from the SeriesDescription column if followed by 'mGy'."""
    import re
    match = re.search(r'(\d+)mGy', series_description)
    if match:
        return int(match.group(1))
    else:
        return None

def define_regressor(input_size):
    def mlp(x, dropout_rate, hidden_units):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    input = tf.keras.Input(shape=(input_size,))
    ff = mlp(input, 0.2, [100, 60, 30])
    
    # Output layer: Linear activation for regression
    output = layers.Dense(1, activation='linear')(ff)

    model = tf.keras.Model(inputs=input, outputs=output)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model

def train_and_test(X_train, y_train, X_test, y_test, input_size):
    model = define_regressor(input_size)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=8,
        epochs=50,
        verbose=1,
        callbacks=[early_stopping]
    )

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_pred.flatten() - y_test))

    plot_loss(history)
    print(f"Test MAE: {mae:.4f}")

    return mae

def run_cross_validation(data_path, output_dir='./results'):
    data, features, labels = load_data(data_path)

    # Extract the method name from the file path
    method_name = os.path.basename(data_path).split('_')[1] 

    groups = data['ROI_Scanner_Pair']  # Group by ROI-Scanner pairs

    group_kfold = GroupKFold(n_splits=10)  # 10 folds
    fold_maes = []

    results_file = os.path.join(output_dir, f'regression_{method_name}.txt')

    with open(results_file, 'w') as file:
        file.write('Fold MAE Results:\n')

        for fold, (train_idx, test_idx) in enumerate(group_kfold.split(features, labels, groups)):
            print(f"\n=== Fold {fold + 1} ===")

            # Train-test split
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Train the model
            mae = train_and_test(X_train, y_train, X_test, y_test, X_train.shape[1])
            fold_maes.append(mae)

            print(f"Fold {fold + 1} MAE: {mae:.4f}")
            file.write(f"Fold {fold + 1} MAE: {mae:.4f}\n")

        # Compute average MAE
        mean_mae = np.mean(fold_maes)
        print(f"\nFinal 10-Fold MAE: {mean_mae:.4f}")
        file.write(f"\nFinal 10-Fold MAE: {mean_mae:.4f}\n")

        # Plot the MAE per fold and mean MAE
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(1, 11), fold_maes, color='skyblue', label="MAE per Fold")
        plt.axhline(mean_mae, color='r', linestyle='--', label=f"Mean MAE: {mean_mae:.4f}")
        plt.xlabel("Fold")
        plt.ylabel("MAE")
        plt.title(f"10-Fold Cross-Validation MAE {method_name}")
        plt.legend()

        # Add numbers above the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

        plt.ylim(0, 4)

        # Save the plot
        plot_path = os.path.join(output_dir, f'regression_{method_name}.png')
        plt.savefig(plot_path)
        print(f"Plot saved at: {plot_path}")

        plt.show()

    return mean_mae


def plot_loss(history):
    """Plots training and validation loss."""
    plt.figure(figsize=(10, 5))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.show()


def main():
    files_dir = '/mnt/nas7/data/maria/final_features'
    #files_dir = '/mnt/nas7/data/maria/final_features/small_roi'
    output_dir = '/mnt/nas7/data/maria/final_features/dose_regression'
    #output_dir = '/mnt/nas7/data/maria/final_features/dose_classification/small_roi'
    os.makedirs(output_dir, exist_ok=True)

    csv_paths = [
        f'{files_dir}/features_pyradiomics_full.csv',
        f'{files_dir}/features_cnn_full.csv',
        f'{files_dir}/features_swinunetr_full.csv',
        f'{files_dir}/features_ct-fm_full.csv'
    ]

    for path in csv_paths:
        run_cross_validation(path, output_dir=output_dir)

if __name__ == '__main__':
    main()

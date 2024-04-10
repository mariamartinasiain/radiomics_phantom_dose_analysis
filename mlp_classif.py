import re
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def group_data(data, mode='repetition'):
    gd = {}
    if mode == 'scanner':
        # Extract the first two characters and map them to unique integers
        unique_groups = data['SeriesDescription'].apply(lambda x: x[:2]).unique()
        group_map = {group: i for i, group in enumerate(unique_groups)}
        gd['group_id'] = data['SeriesDescription'].apply(lambda x: group_map[x[:2]])
    elif mode == 'repetition':
        # Extract the base part excluding the numeric suffix and map them to unique integers
        def extract_base(description):
            base = re.match(r"(.+)(-\s#\d+)$", description)
            if base:
                return base.group(1).strip()
            return description
        
        gd['base'] = data['SeriesDescription'].apply(extract_base)
        unique_bases = gd['base'].unique()
        base_map = {base: i for i, base in enumerate(unique_bases)}
        gd['group_id'] = gd['base'].apply(lambda x: base_map[x])
    
    return np.array(gd['group_id'])

def load_csv(file_path):
    data = pd.read_csv(file_path)

    groups = group_data(data)

    # Standardize ROI labels
    data['ROI'] = data['ROI'].str.replace(r'\d+', '', regex=True)
    labels = data['ROI'].values
    
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    if features.columns[0] == 'deepfeatures':
        features = features['deepfeatures'].apply(eval).apply(pd.Series)
    features = features.values
    
    print(f'Loaded {len(features)} samples with {len(features[0])} features')
    
    return features, labels,groups

def load_data(file_path,one_hot=True):
    scaler = StandardScaler()
    
    features, labels,groups = load_csv(file_path)
    features = scaler.fit_transform(features)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))
    
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    if one_hot:
        labels = to_categorical(labels)
    
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(features, labels, groups=groups))
    x_train, x_val = features[train_idx], features[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    
    train_groups = np.unique(groups[train_idx])
    val_groups = np.unique(groups[val_idx])
    
    print(f'Loaded {len(x_train)} training samples and {len(x_val)} validation samples')
    print(f'Training groups: {train_groups}')
    print(f'Validation groups: {val_groups}')
    
    return x_train, y_train, x_val, y_val,class_weights

def define_classifier(input_size):
    def mlp(x, dropout_rate, hidden_units):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    input = tf.keras.Input(shape=(input_size,))
    ff = mlp(input, 0.1, [250,150, 75])
    classif = layers.Dense(4, activation='softmax')(ff)

    classifier = tf.keras.Model(inputs=input, outputs=classif)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    classifier.summary()
    print(classifier.summary())
    
    return classifier

def save_classifier_performance(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.close()

    
    
def train_classifier(input_size, data_path):
    classifier = define_classifier(input_size)
    x_train, y_train, x_val, y_val,cw = load_data(data_path)

    history = classifier.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=150,
        verbose=2,
        class_weight=cw
    )
    save_classifier_performance(history)
    classifier.save('classifier.h5')
    
def main():
    train_classifier(86, 'data/output/features.csv')
    
if __name__ == '__main__':
    main()

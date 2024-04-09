import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def load_csv(file_path):
    data = pd.read_csv(file_path)

    # Standardize ROI labels
    data['ROI'] = data['ROI'].str.replace(r'\d+', '', regex=True)
    labels = data['ROI'].values
    
    features = features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    features = features.values
    
    print(f'Loaded {len(features)} samples with {len(features[0])} features')
    
    return features, labels

def load_data(file_path):
    scaler = StandardScaler()
    
    features, labels = load_csv(file_path)
    features = scaler.fit_transform(features)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))
    
    x_train, x_val, y_train, y_val = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)
    
    print(f'Loaded {len(x_train)} training samples and {len(x_val)} validation samples')
    
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
        epochs=250,
        verbose=2,
        class_weight=cw
    )
    save_classifier_performance(history)
    classifier.save('classifier.h5')
    
def main():
    train_classifier(86, 'data/output/features.csv')
    
if __name__ == '__main__':
    main()

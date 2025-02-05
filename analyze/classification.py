import sys
sys.path.append('/home/reza/radiomics_phantom/')
sys.path.append('/home/reza/radiomics_phantom/analyze')

from analyze import extract_mg_value
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
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut,LeavePGroupsOut,GroupKFold, KFold
from sklearn import svm
import numpy as np
import os

def group_data(data, mode='scanner'):
    gd = {}
    if mode == 'scanner':
        # Extract the first two characters and map them to unique integers
        unique_groups = data['SeriesDescription'].apply(lambda x: x[:2]).unique()
        #lexicographic order of unique groups
        unique_groups.sort()
        group_map = {group: i for i, group in enumerate(unique_groups)}
        #printing group id  <-> group name mapping
        print({v: k for k, v in group_map.items()})
        gd['group_id'] = data['SeriesDescription'].apply(lambda x: group_map[x[:2]])
    elif mode == 'repetition':
        # Extract the base part excluding the numeric suffix and map them to unique integers
        def extract_base(description):
            base = re.match(r"(.+?)_(IR|FBP|DL)(\s-\s#\d+)$", description)
            if base:
                print(f"base: {base.group(1).strip()}")
                return base.group(1).strip()
            return description
        
        gd['base'] = data['SeriesDescription'].apply(extract_base)
        unique_bases = gd['base'].unique()
        unique_bases.sort()
        base_map = {base: i for i, base in enumerate(unique_bases)}
        #printing group id  <-> group name mapping
        print({v: k for k, v in base_map.items()})
        gd['group_id'] = gd['base'].apply(lambda x: base_map[x])
    
    
    return np.array(gd['group_id'])

def load_csv(file_path, label_type='roi_small',mg_filter=None):
    print(f'Loading data from {file_path}')
    data = pd.read_csv(file_path)
    
    print("mg filter",mg_filter)
    print(f'Length of data before filtering: {len(data)}')
    datad = {}
    datad['mg_value'] = data['SeriesDescription'].apply(extract_mg_value)
    if mg_filter is not None:
        data = data[datad['mg_value'] == mg_filter]
        print(f'Length of data after filtering: {len(data)}')

    print("Grouping data...")
    if label_type == 'scanner':
        groups = group_data(data, mode='repetition')
    else:
        groups = group_data(data)
    print(f'Found {len(np.unique(groups))} unique groups')

    # Standardize ROI labels
    if label_type == 'roi_small':
        data['ROI'] = data['ROI'].str.replace(r'\d+', '', regex=True)
        labels = data['ROI'].values
    elif label_type == 'roi_large':
        data['ROI'] = data['ROI']
        labels = data['ROI'].values
    elif label_type == 'scanner':
        labels = data['SeriesDescription'].str[:2].values
    
    print(f'Found {len(np.unique(labels))} unique labels for {label_type}')
    print(f'Labeled classes: {np.unique(labels)} for {label_type}')
    
    features = data.drop(columns=['StudyInstanceUID', 'SeriesNumber', 'SeriesDescription', 'ROI','ManufacturerModelName','Manufacturer','SliceThickness','SpacingBetweenSlices'],errors='ignore')
    if 'deepfeatures' in data.columns:
        features = features['deepfeatures'].apply(eval).apply(pd.Series)
    features = features.values
    
    print(f'Loaded {len(features)} samples with {len(features[0])} features')
    
    return features, labels,groups

def load_data(file_path,one_hot=True, label_type='roi_small',mg_filter=None):
    print
    scaler = StandardScaler()
    
    features, labels,groups = load_csv(file_path, label_type=label_type,mg_filter=mg_filter)
    features = scaler.fit_transform(features)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    if one_hot:
        labels = to_categorical(labels)
    print(f'Found {len(np.unique(labels))} unique labels')
    print(f'Labeled classes: {label_encoder.classes_}')
    classes_size = len(label_encoder.classes_)
    
    #leave one groupe out (do it as much times as there is scanner(loop is probably BEFORE this call))
    #test will be the group leavead out and train will be the train from splits
    
    
    return features, labels, groups, class_weights, classes_size


def advanced_split(test_manufacuturer, train_size_ratio, X, y, groups, groups_manufacturer, random_state=42):
    # Return the indices where the manufacturer is the same as the test manufacturer
    same_manufacturer = np.where(groups_manufacturer == test_manufacuturer)[0]
    other_manufacturers = np.where(groups_manufacturer != test_manufacuturer)[0]
    number_of_scanners_in_testset = round(train_size_ratio/(1/12))
    number_of_scanners_in_others = round(len(other_manufacturers)/len(groups_manufacturer)*12)
    np.random.seed(random_state//2)
    if number_of_scanners_in_testset <= number_of_scanners_in_others:
        manufacturers = np.unique(groups_manufacturer)
        other_manufacturers = manufacturers.tolist()
        other_manufacturers.remove(test_manufacuturer)
        valid_scanners = np.unique(groups[groups_manufacturer != test_manufacuturer])
        # Randomly select scanners from the valid scanners
        # selected_scanners = np.random.choice(valid_scanners, number_of_scanners_in_testset, replace=False)
        shuffled_scanners = np.random.permutation(valid_scanners)
        selected_scanners = shuffled_scanners[:number_of_scanners_in_testset]
        train_indecies = np.where(np.isin(groups, selected_scanners))[0]
    else:
        valid_scanners = np.unique(groups[groups_manufacturer == test_manufacuturer])
        shuffled_scanners = np.random.permutation(valid_scanners)
        selected_scanners = shuffled_scanners[:number_of_scanners_in_testset-number_of_scanners_in_others]
        # selected_scanners = np.random.choice(valid_scanners, number_of_scanners_in_testset-number_of_scanners_in_others, replace=False)
        selected_scanners = np.concatenate((selected_scanners, np.unique(groups[groups_manufacturer != test_manufacuturer])))
        train_indecies = np.where(np.isin(groups, selected_scanners))[0]

    return [[train_indecies, None]]

def define_classifier(input_size,classes_size):
    def mlp(x, dropout_rate, hidden_units):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    input = tf.keras.Input(shape=(input_size,))
    ff = mlp(input, 0.2, [100,60, 30])
    classif = layers.Dense(classes_size, activation='softmax')(ff)

    classifier = tf.keras.Model(inputs=input, outputs=classif)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # classifier.summary()
    # print(classifier.summary())
    
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

def train_mlp_svm(input_size, data_path, output_path='classifier.h5', classif_type='roi_small', mg_filter=None):
    features, labels, groups, cw, classes_size = load_data(data_path, label_type=classif_type, mg_filter=mg_filter)
    
    val_accuracies = []
    min_accuracy, max_accuracy = 1, 0
    results = {}

    save_results_to_csv([])

    # Use GroupKFold for 'scanner' classification, otherwise LeaveOneGroupOut
    if classif_type == 'scanner':
        split_strategy = KFold(n_splits=10, shuffle=True, random_state=42).split(features, labels)
        # split_strategy = GroupKFold(n_splits=10).split(features, labels, groups)
    else:
         split_strategy = LeaveOneGroupOut().split(features, labels, groups)
        #  split_strategy = KFold(n_splits=10, shuffle=True, random_state=42).split(features, labels) # For 10-fold ROI classification

    scanner_to_manufacturer = {0: 0, 1: 0, 2: 0, 3: 0, 9: 0, 10: 0,
                                   4:1, 12:1, 5:2, 7:2, 8:2, 6:3, 11:3}
    groups_manufacturer = np.array([scanner_to_manufacturer[scanner] for scanner in groups])

    # Iterate over each train-test split
    for scanner_id, (train_idx, test_idx) in enumerate(split_strategy):
        if 'loso' in data_path and scanner_id != 0:
            features, labels, groups, cw, classes_size = load_data(data_path.replace(f'00', f'{scanner_id :02d}'),
                                                                    label_type=classif_type, mg_filter=mg_filter)
            groups_manufacturer = np.array([scanner_to_manufacturer[scanner] for scanner in groups])
    
        X_train_all, X_test = features[train_idx], features[test_idx]
        y_train_all, y_test = labels[train_idx], labels[test_idx]
        groups_train_all = groups[train_idx]
        unique_train_groups = np.unique(groups_train_all)
        groups_train_manufacturer_all = groups_manufacturer[train_idx]

        if classif_type == 'scanner':
            it2 = range(1, 10)
        else:
            it2 = range(1, len(unique_train_groups) + 1)
            # it2 = range(1, 10) # For 10-fold ROI classification

        for N in it2:
            train_size_ratio = N / (10 if classif_type == 'scanner' else len(unique_train_groups))
            # train_size_ratio = N / 10 # For 10-fold ROI classification
            if train_size_ratio == 1:
                it3 = [(np.arange(len(X_train_all)), None)]
            else:
                it3 = advanced_split(scanner_to_manufacturer[scanner_id], train_size_ratio, X_train_all, y_train_all, groups_train_all, groups_train_manufacturer_all, random_state=42)
                # it3 = [(np.arange(len(X_train_all)), None)] # For 10-fold ROI classification
                #it3 = GroupShuffleSplit(n_splits=1, train_size=train_size_ratio, random_state=42).split(X_train_all, y_train_all, groups_train_all)

            for train_indices, _ in it3:
                X_train, y_train = X_train_all[train_indices], y_train_all[train_indices]
                print("\033[94mData splits:\033[0m", scanner_to_manufacturer[scanner_id], np.unique(groups_train_manufacturer_all[train_indices]), np.unique(groups_train_all[train_indices]))
                # Define and train the classifier
                classifier = define_classifier(input_size, classes_size)
                history = classifier.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    batch_size=8,
                    epochs=30,
                    verbose=1,
                    class_weight=cw
                )
                
                y_pred = classifier.predict(X_test)
                val_accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) 
                val_accuracies.append(val_accuracy)

                results.setdefault(N, []).append(val_accuracy)

                print(f"Test group: {test_idx + 1}, Training with N={N}, Accuracy: {val_accuracy}")                

    # Compute overall results
    mean_val_accuracy = np.mean(val_accuracies)
    min_accuracy = np.min(val_accuracies)
    max_accuracy = np.max(val_accuracies)
    print(f"list of accuracies: {val_accuracies}")
    print(f"Final results: Mean accuracy: {mean_val_accuracy}, Min accuracy: {min_accuracy}, Max accuracy: {max_accuracy}")
    save_results_to_csv(results, classif_type=classif_type, mg_filter=mg_filter, data_path=data_path, plus="9999")

    return val_accuracies, max_accuracy, min_accuracy
    
def save_results_to_csv(results,classif_type='roi_small',mg_filter=None,data_path="",plus=""):
    df = pd.DataFrame(results)
    
    base_name = os.path.basename(data_path)
    base_name = os.path.splitext(base_name)[0]
    
    #adding columns
    df['classif_type'] = classif_type
    df['mg_filter'] = mg_filter
    csv_filename = f"results_{classif_type}_{plus}_{mg_filter}_{base_name}.csv"

    # directory = os.path.dirname(csv_filename)
    # if directory and not os.path.exists(directory):
    #     os.makedirs(directory)
    print(f"Saved results to {csv_filename}")
    
    os.makedirs("./results", exist_ok=True)
    df.to_csv(os.path.join("./results",csv_filename), index=False)  
    # os.makedirs("./results_combat", exist_ok=True)
    # df.to_csv(os.path.join("./results_combat",csv_filename), index=False)
    # os.makedirs("./results_10fold", exist_ok=True) # For 10-fold ROI classification
    # df.to_csv(os.path.join("./results_10fold",csv_filename), index=False) # For 10-fold ROI classification 
 
def train_mlp_with_data(x_train, y_train, x_val, y_val, input_size, output_path='classifier.h5'):
    classifier = define_classifier(input_size)
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = dict(enumerate(cw))
    
    history = classifier.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=70,
        verbose=2,
        class_weight=cw
    )
    save_classifier_performance(history)
    classifier.save(output_path)
    max_val_accuracy = max(history.history['val_accuracy'])
    return max_val_accuracy
    
def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid taking all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Using CPU.")
    train_mlp_svm(86, 'data/output/features.csv')
    
if __name__ == '__main__':
    main()

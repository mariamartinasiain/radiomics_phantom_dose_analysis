import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_data_for_classification(filepath):
    """Loads data and prepares features and labels for classification."""
    data = pd.read_csv(filepath)

    if 'deepfeatures' in data.columns and data['deepfeatures'].dtype == 'object':
        data['deepfeatures'] = data['deepfeatures'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        max_len = data['deepfeatures'].apply(len).max()
        feature_df = pd.DataFrame(data['deepfeatures'].tolist(), index=data.index)
        feature_df.columns = [f"feature_{i}" for i in range(max_len)]
        data = pd.concat([data.drop(columns=['deepfeatures']), feature_df], axis=1)

    features = data[[col for col in data.columns if col.startswith("feature_")]]
    labels = data['Dose'].astype(str)

    return features, labels

def classify_dose(features, labels):
    """Performs binary classification between Full Dose and Quarter Dose."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return clf

if __name__ == "__main__":
    filepath = '/mnt/nas7/data/maria/final_features/ct-fm_low_dose/features_ct-fm_low_dose_v2.csv'

    features, labels = load_data_for_classification(filepath)
    classify_dose(features, labels)

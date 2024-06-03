## Step 1: Recognize the pilots' intention with Machine Learning models
## Train an GBDT model and test its performance using 5-fold cross-validation (5 times)


import os
import json
import csv
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def extract_features(data):
    """ Converts raw JSON data into a fixed-length feature vector, including pupil size. """
    features = {}
    for entry in data:
        for key, value in entry.items():
            if value is None:
                continue  # Skip None values
            if key == 'Pupil':
                # Handle pupil size separately
                pupil_key = 'Pupil'
                if pupil_key in features:
                    features[pupil_key].append(value)
                else:
                    features[pupil_key] = [value]
            else:
                # Extract duration features
                if key in features:
                    features[key].append(value)
                else:
                    features[key] = [value]

    # Calculate statistical features for each key
    feature_vector = []
    for key, values in features.items():
        if not values:
            continue  # Skip empty lists
        feature_vector.append(np.mean(values))  # Mean
        feature_vector.append(np.std(values))   # Standard Deviation
        feature_vector.append(np.min(values))   # Min
        feature_vector.append(np.max(values))   # Max
    return np.array(feature_vector)

def load_data(base_directory):
    """ Load data from specified directory and return feature vectors with labels """
    data = []
    folders = ['Pos', 'Neg']  # Define folder names corresponding to labels
    label_mapping = {'Pos': 1, 'Neg': 0}  # Map folder names to binary labels
    for folder in folders:
        directory = os.path.join(base_directory, folder)
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    content = json.load(file)
                    features = extract_features(content)
                    binary_label = label_mapping[folder]
                    data.append((features, binary_label, folder))
    return data

def cross_validate(data, n_splits=5, n_repeats=5):
    """ Perform cross-validation to evaluate model performance """
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    feature_vectors = [entry[0] for entry in data]
    labels = [entry[1] for entry in data]
    folders = [entry[2] for entry in data]

    # Find the maximum length of the feature vectors
    max_length = max(len(f) for f in feature_vectors)
    # Pad all feature vectors to the maximum length
    padded_feature_vectors = [np.pad(f, (0, max_length - len(f)), 'constant') for f in feature_vectors]

    for _ in range(n_repeats):
        for train_index, test_index in kf.split(padded_feature_vectors):
            train_data = [padded_feature_vectors[i] for i in train_index]
            train_labels = [labels[i] for i in train_index]
            test_data = [padded_feature_vectors[i] for i in test_index]
            test_folders = [folders[i] for i in test_index]

            # Train the Gradient Boosting model
            gbdt_model = GradientBoostingClassifier()
            gbdt_model.fit(train_data, train_labels)
            predicted_labels = gbdt_model.predict(test_data)

            for folder, predicted in zip(test_folders, predicted_labels):
                results.append({'Actual': folder, 'Predicted': str(int(predicted))})

    return results

def save_results(results, output_path):
    """ Save the cross-validation results to a CSV file """
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Actual', 'Predicted'])
        writer.writeheader()
        writer.writerows(results)

def main():
    """ Main function to orchestrate data loading, model training, and result saving """
    base_directory = '/workspaces/Mr-LLM/Data/Traces/10'
    output_path = '/workspaces/Mr-LLM/Data/Traces/10/GBDT-10.csv'

    data = load_data(base_directory)
    results = cross_validate(data)
    save_results(results, output_path)

if __name__ == "__main__":
    main()

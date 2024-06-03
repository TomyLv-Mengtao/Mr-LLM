## Step 1: Recognize the pilots intention with Machine Learning models
## Train an HMM model and test its performance using 5-fold cross-validation (5 times)

# pip install xgboost
import os
import json
import csv
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb

def extract_features(data):
    """ Converts raw JSON data into a fixed-length feature vector. """
    features = {}
    for entry in data:
        for key, value in entry.items():
            if value is None:
                continue  # Skip None values
            if key in features:
                features[key].append(float(value))
            else:
                features[key] = [float(value)]
    # Calculate statistical features for each key
    feature_vector = []
    for key, values in features.items():
        if values:  # Ensure the list is not empty
            feature_vector.append(np.mean(values))  # Mean
            feature_vector.append(np.std(values))   # Standard Deviation
            feature_vector.append(np.min(values))   # Min
            feature_vector.append(np.max(values))   # Max
    return np.array(feature_vector)

def load_data(base_directory):
    """ Load data from specified directory and return feature vectors with labels """
    data = []
    folders = ['Pos', 'Neg']  # Map folder names to binary labels
    label_mapping = {'Pos': 1, 'Neg': 0}
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
    """ Perform cross-validation using XGBoost for model training and evaluation """
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    for _ in range(n_repeats):
        for train_index, test_index in kf.split(data):
            train_data = [data[i][0] for i in train_index]
            train_labels = [data[i][1] for i in train_index]
            test_data = [data[i][0] for i in test_index]
            test_folders = [data[i][2] for i in test_index]

            # Pad data to ensure all inputs are the same length
            max_length = max(len(x) for x in train_data + test_data)
            train_data_padded = [np.pad(x, (0, max_length - len(x)), 'constant') for x in train_data]
            test_data_padded = [np.pad(x, (0, max_length - len(x)), 'constant') for x in test_data]

            # Create XGBoost DMatrices
            dtrain = xgb.DMatrix(train_data_padded, label=train_labels)
            dtest = xgb.DMatrix(test_data_padded)

            # XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 4,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }

            # Train the model
            bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train')], early_stopping_rounds=10)

            # Predicting using the best iteration
            predicted_labels = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
            predicted_labels = (predicted_labels >= 0.5).astype(int)  # Convert probabilities to binary output

            for folder, predicted in zip(test_folders, predicted_labels):
                results.append({'Actual': folder, 'Predicted': str(predicted)})

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
    output_path = '/workspaces/Mr-LLM/Data/Traces/10/XGB-10.csv'

    data = load_data(base_directory)
    results = cross_validate(data)
    save_results(results, output_path)

if __name__ == "__main__":
    main()

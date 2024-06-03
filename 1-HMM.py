## Step 1: Recognize the pilots intention with Machine Learning models
## Train an HMM model and test its performance using 5-fold cross-validation (5 times)

# pip install hmmlearn
import os
import json
import csv
from sklearn.model_selection import KFold
import numpy as np
from hmmlearn import hmm

class HiddenMarkovModel:
    def __init__(self, n_components=3, n_iter=100):
        # Initialize a Multinomial Hidden Markov Model with a given number of states and iterations
        self.model = hmm.MultinomialHMM(n_components=n_components, n_iter=n_iter)

    def train(self, data):
        # Prepare data for training: extract sequences and their lengths
        lengths = [len(seq) for seq, _ in data]
        combined_seq = np.concatenate([seq for seq, _ in data])
        # Fit the HMM model using the combined sequences and their respective lengths
        self.model.fit(combined_seq.reshape(-1, 1), lengths=lengths)

    def predict(self, seq):
        # Predict the state sequence using the model and return the most likely state
        logprob, states = self.model.decode(np.array(seq).reshape(-1, 1))
        return np.bincount(states).argmax()

def encode_data(data):
    # Generate a unique integer ID for each unique event in the data
    unique_items = set()
    for seq, _ in data:
        unique_items.update(seq)
    item_to_id = {item: idx for idx, item in enumerate(unique_items)}
    # Encode data sequences to integer IDs for model compatibility
    encoded_data = [(np.array([item_to_id[item] for item in seq], dtype=int), label) for seq, label in data]
    return encoded_data, item_to_id

def load_data(base_directory):
    # Load data from specified directory and return encoded sequences with labels
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
                    flattened_content = flatten(content)
                    binary_label = label_mapping[folder]
                    data.append((flattened_content, binary_label, folder))
    return data

def flatten(data):
    # Flatten JSON data into a format suitable for encoding and training
    if isinstance(data, dict):
        event = next(iter(data))
        duration = data[event]
        pupil_size = data.get('Pupil', 1)  # Use default pupil size of 1 if not specified
        return [(event, duration, pupil_size)]
    elif isinstance(data, list):
        return sum([flatten(item) for item in data], [])
    else:
        return [data]

def cross_validate(data, n_splits=5, n_repeats=5):
    # Perform cross-validation to evaluate model performance
    results = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    encoded_data, _ = encode_data([(seq, _) for seq, _, __ in data])
    for _ in range(n_repeats):
        for train_index, test_index in kf.split(encoded_data):
            train_data = [encoded_data[i] for i in train_index]
            test_data = [encoded_data[i] for i in test_index]
            test_folders = [data[i][2] for i in test_index]

            hmm_model = HiddenMarkovModel(n_components=len(set([label for _, label, _ in data])))
            hmm_model.train(train_data)

            for (seq, _), folder in zip(test_data, test_folders):
                predicted_type = hmm_model.predict(seq)
                results.append({'Actual': folder, 'Predicted': predicted_type})
    return results

def save_results(results, output_path):
    # Save the cross-validation results to a CSV file
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Actual', 'Predicted'])
        writer.writeheader()
        for result in results:
            result['Predicted'] = '1' if result['Predicted'] == 1 else '0'
            writer.writerow(result)


def main():
    """ Main function to orchestrate data loading, model training, and result saving """
    base_directory = '/workspaces/Mr-LLM/Data/Traces/10'
    output_path = '/workspaces/Mr-LLM/Data/Traces/10/HMM-10.csv'

    data = load_data(base_directory)
    results = cross_validate(data)
    save_results(results, output_path)

if __name__ == "__main__":
    main()

## Step 1: Recognize the pilots intention with Machine Learning models
## Train an HMM model and test its performance using 5-fold cross-validation (5 times)

# pip install tensorflow

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Paths to the positive and negative JSON files
positive_path = '/workspaces/Mr-LLM/Data/Traces/2/Pos'
negative_path = '/workspaces/Mr-LLM/Data/Traces/2/Neg'


# Define a function to convert AOI to numerical values
def aoi_to_num(aoi):
    mapping = {
        'FCU': 0,
        'PFD': 1,
        'ND': 2,
        'ECAM and MFD': 3,
        'LDG': 4,
        'MCDU': 5,
        'External': 6,
        'Others': 7,
        'saccade': 8
    }
    return mapping.get(aoi, 9)  # Default to 9 if not found

# Function to load data from a directory
def load_data_from_directory(path, label):
    data = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            with open(os.path.join(path, filename)) as f:
                trace = json.load(f)
                sequence = []
                for event in trace:
                    for aoi, duration in event.items():
                        if aoi != "Pupil":
                            pupil_size = event["Pupil"]
                            aoi_num = aoi_to_num(aoi)
                            sequence.append([aoi_num, duration, pupil_size])
                data.append(sequence)
                labels.append(label)
    return data, labels

# Load data
pos_data, pos_labels = load_data_from_directory(positive_path, 1)
neg_data, neg_labels = load_data_from_directory(negative_path, 0)

# Combine data
data = pos_data + neg_data
labels = np.array(pos_labels + neg_labels)

# Pad sequences
maxlen = max(len(seq) for seq in data)
X = pad_sequences(data, padding='post', dtype='float32', maxlen=maxlen)

# Define the LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5-fold cross-validation repeated 5 times
n_splits = 5
n_repeats = 5
kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)

results = []
actuals = []
predictions = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    model = create_model((X.shape[1], X.shape[2]))

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()

    actuals.extend(y_test)
    predictions.extend(y_pred)

# Convert actual labels to "Pos" and "Neg"
actuals = ["Pos" if label == 1 else "Neg" for label in actuals]

# Save the results to a CSV file
output_df = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})
output_df.to_csv('/workspaces/Mr-LLM/Data/Traces/2/LSTM.csv', index=False)

print("Cross-validation complete. Results saved to output.csv.")

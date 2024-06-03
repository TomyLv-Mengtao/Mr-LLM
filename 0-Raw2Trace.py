## Step 0: Raw data from the Tobii Pro Lab "Data Export" function, encode the data into Visual Attention Traces (VATs)
## Filter: Tobii I-VT (Attention)


import os
import pandas as pd
import json
from datetime import timedelta
import re
import random

# Base directory for the data
base_directory = '/workspaces/Mr-LLM/Data/Raw'  # An example file is provided
output_base = '/workspaces/Mr-LLM/Data/Traces/2'

# Paths for each task's output
output_paths = {
    'Task1-Detected': os.path.join(output_base, 'Pos'),
    'Task2-Detected': os.path.join(output_base, 'Pos'),
    'Task2-RandomSlice': os.path.join(output_base, 'Neg'),  # For randomly selected Neg traces
}

# Ensure all output directories exist
for path in output_paths.values():
    os.makedirs(path, exist_ok=True)

def simplify_aoi_name(aoi_column_name):
    match = re.search(r'\[(.*)\]', aoi_column_name)
    return match.group(1).split(' - ')[-1] if match else "Unknown"

def process_file(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['Timestamp'] = pd.to_datetime(df['Recording timestamp [μs]'], unit='us')
    # Exclude rows where pupil diameter is NaN
    pupil_diameters = df['Pupil diameter filtered [mm]'].dropna()
    # Calculate the average pupil diameter
    overall_pupil_diameter = pupil_diameters.mean()

    # Define time intervals
    END = 0
    START = 2
    TIME_WINDOW = END + START

    # Events to find with their respective output directories
    tasks = ['Task1-Detected', 'Task2-Detected']
    task_times = {}

    # Find event times
    for task in tasks:
        try:
            task_times[task] = df[df['Event'] == task]['Timestamp'].min()
        except ValueError:
            pass  # Skip if the task is not found

    # Process regular detected tasks
    for task in tasks:
        if task in task_times:
            start_time = task_times[task] - timedelta(seconds = START)
            end_time = task_times[task] + timedelta(seconds = END)
            filtered_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]
            save_trace(filtered_df, task, file_path, overall_pupil_diameter)

    # Randomly select slices for Task2 if detected
    if 'Task2-Detected' in task_times:
        start_time = task_times['Task2-Detected'] + timedelta(minutes=5)
        end_time = task_times['Task2-Detected'] + timedelta(minutes=15)
        # Generate four random start times for duration slices
        for i in range(2):
            random_start = start_time + timedelta(seconds=random.randint(0, 600 - TIME_WINDOW))
            random_end = random_start + timedelta(seconds = TIME_WINDOW)
            filtered_df = df[(df['Timestamp'] >= random_start) & (df['Timestamp'] <= random_end)]
            save_trace(filtered_df, 'Task2-RandomSlice', file_path, overall_pupil_diameter, slice_index=i+1)

def save_trace(filtered_df, key, file_path, overall_pupil_diameter, slice_index=None):
    trace = []
    grouped = filtered_df.groupby((filtered_df['Eye movement type'] != filtered_df['Eye movement type'].shift(1)).cumsum())
    for _, group in grouped:
        first_row = group.iloc[0]
        event_type = first_row['Eye movement type']
        duration = int(group['Gaze event duration [ms]'].iloc[0])
        pupil_diameters = group['Pupil diameter filtered [mm]'].dropna()
        normalized_pupil = round(pupil_diameters.mean() / overall_pupil_diameter, 2) if not pupil_diameters.empty else None

        if event_type == 'Fixation':
            aoi_columns = group.filter(regex='^AOI hit').columns
            aoi_hits = {simplify_aoi_name(col): group[col].sum() for col in aoi_columns}
            aoi_name = next((name for name, hit in aoi_hits.items() if hit > 0), 'Others')
            trace.append({aoi_name: duration, 'Pupil': normalized_pupil})
        elif event_type == 'Saccade':
            trace.append({'saccade': duration, 'Pupil': normalized_pupil})

    # Generate a unique filename for each slice
    suffix = f'_{key}'
    if slice_index is not None:
        suffix += f'_Slice{slice_index}'
    output_file = os.path.join(output_paths[key], os.path.basename(file_path).replace('.tsv', f'{suffix}.json'))

    with open(output_file, 'w') as f:
        json.dump(trace, f, indent=4)
    print(f"Trace for {key}{' ' if slice_index is None else ' slice ' + str(slice_index)} saved to: {output_file}")

# Process all TSV files in the directory
for filename in os.listdir(base_directory):
    if filename.endswith('.tsv'):
        file_path = os.path.join(base_directory, filename)
        process_file(file_path)

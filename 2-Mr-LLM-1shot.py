## Step 2: Recognize the pilots intention with Mind-reading based on LLM (Mr-LLM)
## One shot of LLM method

import json
import openai
from openai import OpenAI
import os
from pathlib import Path
import csv
import random

# Assistant for GPT- Assistant
class Assistant:
    def __init__(self, model="gpt-4o", knowledge_path=None, files_directory=None):
        self.model = model
        self.api_key = YOUR_API_KEY
        self.client = OpenAI(api_key=self.api_key)

        self.knowledge_path = knowledge_path
        self.files_directory = files_directory

        self.knowledge = self.load_file(self.knowledge_path)
        self.files = [os.path.join(self.files_directory, f) for f in os.listdir(self.files_directory) if os.path.isfile(os.path.join(self.files_directory, f))]

        # Create or load assistant with required files
        self.assistant_id = self.create_or_load_assistant(name="Mr-LLM-1shot", instructions=self.knowledge, model=model)

    def load_file(self, file_path):
        if file_path and Path(file_path).exists():
            with open(file_path, 'r') as file:
                return file.read()
        return ""

    def create_or_load_assistant(self, name, instructions, model):
        assistant_file_path = "assistant.json"
        assistant_json = []

        # Check if the assistant already exists
        if os.path.exists(assistant_file_path):
            with open(assistant_file_path, "r") as file:
                assistant_json = json.load(file)
                for assistant_data in assistant_json:
                    if assistant_data["assistant_name"] == name:
                        print("Loaded the existing Assistant ID: ", assistant_data["assistant_id"])
                        return assistant_data["assistant_id"]

        # Upload files and create Vector Store
        vector_store = self.client.beta.vector_stores.create(name=f"{name} Knowledge Base")
        file_streams = [open(path, "rb") for path in self.files if Path(path).exists()]
        file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )

        # Create a new Assistant if not exist
        assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )

        # Save Assistant info
        assistant_json.append({"assistant_name": name, "assistant_id": assistant.id})
        with open(assistant_file_path, "w", encoding="utf-8") as file:
            json.dump(assistant_json, file, ensure_ascii=False, indent=4)
        print("New Assistant Info saved")

        return assistant.id

    def get_response(self, prompt):
        response = self.client.chat.completions.create(
            model = "gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def send_reference_trace(self, gaze_data):
        prompt = self.generate_reference_prompt(gaze_data)
        # Send reference trace without expecting a response
        self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

    def predict_AD(self, gaze_data):
        prompt = self.generate_prediction_prompt(gaze_data)
        return self.get_response(prompt)

    def generate_reference_prompt(self, gaze_data):
        prompt = f"This is a positive trace example (the pilot is conducting abnormality detection) from a new participant: {gaze_data}\n"
        prompt += "Please use this as a reference for better estimating the traces given later."
        return prompt

    def generate_prediction_prompt(self, gaze_data):
        prompt = f" This is a gaze trace from the same participant: {gaze_data}\n"
        prompt += '''\
            Based on the participant's reference provided earlier, please estimate if the pilot is conducting abnormality detection with this given gaze trace.
            Please organize your answer in a JSON object containing the following keys:
            "Prediction" ("1"/ "0", for your estimation. Reply "1" if you deem the pilot is conducting abnormality detection behaviour. Otherwise, reply "0" if you deem the pilot is just condcuting normal monitoring),
            and "reason" (a concise explanation that supports your estimation according to the requirements). Do not include line breaks in your output
            '''
        return prompt

# Functions for file processing
def clean_json_string(predictions):
    # Remove possible markdown formatting errors
    predictions = predictions.strip('`')

    # Remove leading non-JSON compliant substrings like "json" that precede the actual JSON object
    if predictions.startswith('json'):
        predictions = predictions[4:].strip()

    # Additional cleaning if there are still issues (strip leading whitespaces/newlines)
    predictions = predictions.lstrip()

    return predictions

def group_traces_by_participant(directory):
    grouped_traces = {}
    for folder in ['Pos', 'Neg']:
        for filename in os.listdir(os.path.join(directory, folder)):
            if filename.endswith('.json'):
                # Extract participant ID based on the new simplified format
                participant_id = filename.split('-')[0]
                if participant_id not in grouped_traces:
                    grouped_traces[participant_id] = {'Pos': [], 'Neg': []}
                grouped_traces[participant_id][folder].append(os.path.join(directory, folder, filename))
    return grouped_traces


def process_trace(trace, assistant, actual_label):
    try:
        with open(trace, 'r') as file:
            data = json.load(file)
            trace_data = json.dumps(data)
            prediction = assistant.predict_AD(trace_data)
            clean_predictions = clean_json_string(prediction)
            predictions_dict = json.loads(clean_predictions)
            return {
                'filename': os.path.basename(trace),
                'Actual': actual_label,
                'Prediction': predictions_dict.get('Prediction', 'N/A'),
                'reason': predictions_dict.get('reason', 'N/A')
            }

    except json.JSONDecodeError:
        print('JSONDecodeError for file:', trace)
        return {'filename': os.path.basename(trace), 'Actual': actual_label, 'error': 'JSONDecodeError'}
    except Exception as e:
        print('Error processing file:', trace)
        print('Exception:', str(e))
        return {'filename': os.path.basename(trace), 'Actual': actual_label, 'error': str(e)}

def process_traces_for_participant(participant_traces, assistant):
    results = []
    pos_traces = participant_traces['Pos'].copy()
    neg_traces = participant_traces['Neg'].copy()
    
    # Ensure different example traces for the two rounds
    example_traces = random.sample(pos_traces, 2)

    for round_num in range(2):
        example_trace = example_traces[round_num]
        
        with open(example_trace, 'r') as file:
            example_data = json.load(file)
            example_trace_data = json.dumps(example_data)
            assistant.send_reference_trace(example_trace_data)

        remaining_pos_traces = [trace for trace in pos_traces if trace != example_trace]
        all_traces = remaining_pos_traces + neg_traces
        random.shuffle(all_traces)

        for trace in all_traces:
            actual_label = 'Pos' if 'Detected' in os.path.basename(trace) else 'Neg'
            result = process_trace(trace, assistant, actual_label)
            results.append(result)
            print('Processed file:', trace)
    return results

def main():
    base_directory = '/workspaces/Mr-LLM/Data/Traces/8'

    Knowledge = '/workspaces/Mr-LLM/Knowledge-CoT.txt'
    files_directory = '/workspaces/Mr-LLM/Data/Files'  # The files to be used for file search function of GPT

    assistant = Assistant(model="gpt-4o", knowledge_path= Knowledge, files_directory=files_directory)
    all_results = []

    grouped_traces = group_traces_by_participant(base_directory)

    for participant_id, traces in grouped_traces.items():
        results = process_traces_for_participant(traces, assistant)
        all_results.extend(results)

    # Writing results to CSV
    output_path = '/workspaces/Mr-LLM/Data/Traces/8/MR-1shot-8.csv'
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['filename', 'Actual', 'Prediction', 'reason'])
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print("Results saved to", output_path)


if __name__ == "__main__":
    main()
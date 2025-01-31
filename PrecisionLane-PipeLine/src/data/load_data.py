import pandas as pd
import os

def load_data(raw_data_path):
    # Assuming CSV files for example
    data_files = [os.path.join(raw_data_path, file) for file in os.listdir(raw_data_path)]
    data = pd.concat([pd.read_csv(file) for file in data_files])
    return data

from pathlib import Path

import pandas as pd


class DataLoader:
    def __init__(self):
        self.data = None
        self.file_path = Path(__file__).with_name("bank.csv")

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def file_exists(self, file_path):
        return Path(file_path).is_file()

    def get_file_path(self, file_path):
        if self.file_exists(file_path):
            self.file_path = Path(file_path)
            return self.file_path
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def get_data(self):
        if self.data is not None:
            return self.data
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")
    
    def get_info(self):
        if self.data is not None:
            data_info = {
                "shape": self.data.shape,
                "columns": self.data.columns.tolist(),
                "missing_values": self.data.isnull().sum().to_dict(),
                "data_types": self.data.dtypes.to_dict(),
                "data_info": self.data.info()
            }
            return data_info
        else:
            raise ValueError("Data not loaded. Please load data first using load_data() method.")

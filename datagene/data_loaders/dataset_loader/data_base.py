import os
import json

from abc import *

class DataBase(metaclass=ABCMeta):
    def __init__(self, dataset_path):
        self.dump = None
        
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"{dataset_path} dataset file could not be found")
        
        with open(dataset_path, "r", encoding="utf-8") as fp:
            self.dump = json.load(fp)
            
        if not self.dump:
            raise ValueError("Dataset is empty")
    
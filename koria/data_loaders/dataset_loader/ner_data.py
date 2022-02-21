import os
import json
from tqdm import tqdm

from . import DataBase

class NerData(DataBase):
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        for document in tqdm(self.dump["document"], desc=f"Load dataset"):
            sentences = document["sentence"]
            
            for sentence in sentences:
                form = sentence["form"]
                ne_list = sentence["NE"]
                
                self.data.append({
                    "sentence": form,
                    "entities": ne_list,
                })
                   
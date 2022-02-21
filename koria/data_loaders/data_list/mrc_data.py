import os
import json
from tqdm import tqdm

from . import DataBase

class MrcData(DataBase):
    
    """
    Machine Reading Comprehension Dataset
    - KorQuad 1.0
    """
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        for data in tqdm(self.dump["data"], desc=f"Load dataset"):
            print(data)
            # sentences = document["sentence"]
            
            # for sentence in sentences:
            #     form = sentence["form"]
            #     ne_list = sentence["NE"]
                
            #     self.data.append({
            #         "sentence": form,
            #         "entities": ne_list,
            #     })
                   
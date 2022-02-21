import os
from tqdm import tqdm

from . import DataBase

class SrlData(DataBase):
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        self.data = []
            
        for document in tqdm(self.dump["document"], desc=f"Load dataset"):
            sentences = document["sentence"]
            
            for sentence in sentences:
                form = sentence["form"]
                srl_list = sentence["SRL"]
                
                for srl in srl_list:
                    if "predicate" in srl and "argument" in srl:
                        
                        predicate = srl["predicate"]
                        arguments = srl["argument"]
                        
                        self.data.append({
                            "sentence": form,
                            "predicate": predicate,
                            "arguments": arguments
                        })

    
import os
import json
from tqdm import tqdm

# from . import DataBase
from koria.data_loaders.data_list.data_base import DataBase

class MrcData(DataBase):
    
    """
    Machine Reading Comprehension Dataset
    - KorQuad 1.0
    """
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        for data in tqdm(self.dump["data"], desc=f"Load dataset"):
            
            for paragraph in data["paragraphs"]:
                context = paragraph["context"]        
                
                for qas in paragraph["qas"]:
                    question = qas["question"]
                    answers = qas["answers"]
                    
                    self.data.append({
                        "context": context,
                        "question": question,
                        "answers": answers[0]
                    })

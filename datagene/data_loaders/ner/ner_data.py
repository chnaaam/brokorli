import os
import json
from tqdm import tqdm


class NerData:
    def __init__(self, dataset_dir, dataset_fn):
        self.data = []
        
        dataset_full_path = os.path.join(dataset_dir, dataset_fn)
        if not os.path.isfile(dataset_full_path):
            raise FileNotFoundError(f"{dataset_full_path} dataset file could not be found")
        
        with open(dataset_full_path, "r", encoding="utf-8") as fp:
            dump = json.load(fp)
            
        for document in tqdm(dump["document"], desc=f"Load dataset [{dataset_fn}]"):
            sentences = document["sentence"]
            
            for sentence in sentences:
                form = sentence["form"]
                ne_list = sentence["NE"]
                
                self.data.append({
                    "sentence": form,
                    "entities": ne_list,
                })
                   
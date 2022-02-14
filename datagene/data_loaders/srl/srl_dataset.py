import os
import pickle
from torch.utils.data import Dataset

from .srl_data import SrlData

class SrlDataset(Dataset):
    def __init__(self, tokenizer, dataset_dir, dataset_fn, dataset_type="train", cache_dir=None, vocab_dir=None):
        super().__init__()
        
        self.data = []
        self.tokenizer = tokenizer
        
        cache_full_path = os.path.join(cache_dir, f"{dataset_type}-data.cache")
        
        if not os.path.isfile(cache_full_path):
            # Load dataset from SRL data file and save cache and vocab files
            srl_data = SrlData(dataset_dir=dataset_dir, dataset_fn=dataset_fn)
            
            for data in srl_data:
                sentence = data["sentence"]
                predicate = data["predicate"]
                arguments = data["arguments"]
                
        else:
            # Load cache and vocab files
            pass
        
        
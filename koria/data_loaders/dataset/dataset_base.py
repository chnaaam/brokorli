import os
import pickle

from abc import *
from tqdm import tqdm

from torch.utils.data import Dataset

class DatasetBase(Dataset, metaclass=ABCMeta):
    def __init__(
        self, 
        tokenizer, 
        task_name, 
        model_name, 
        data_list, 
        cache_dir, 
        vocab_dir, 
        build_dataset_func, 
        dataset_type="train", 
        max_seq_len=256, 
        special_tokens=None):
        
        super().__init__()
        
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.model_name = model_name
        self.data_list = data_list
        
        self.cache_dir = cache_dir
        self.vocab_dir = vocab_dir
        
        self.dataset_type = dataset_type
        self.max_seq_len = max_seq_len
        
        self.special_tokens = special_tokens
        
        if self.model_name == "bert":
            self.special_tokenizer_sep_indicator = "▁"
            self.special_tokenizer_replaced_sep_indicator = " "
        else:
            self.special_tokenizer_sep_indicator = "##"
            self.special_tokenizer_replaced_sep_indicator = ""
        
        self.build_dataset_func = build_dataset_func
        
        self.dataset = []
        
        self.build()
        
    def build(self):
        # We use cache file for improving data loading speed when cache file is existed in cache directory.
        # But file is not existed, then build dataset and save into cache file
        cache_path = os.path.join(self.cache_dir, f"{self.dataset_type}-{self.task_name}-{self.model_name}-data.cache")
        
        if not os.path.isfile(cache_path):
            for data in tqdm(self.data_list, desc=f"Load {self.dataset_type} data"):
                tokenized_data = self.build_dataset_func(data)
                
                if tokenized_data:
                    self.dataset.append(tokenized_data)
                
            self.save_cache_file(cache_path, self.dataset)
        else:
            self.dataset = self.load_cache_file(cache_path)
    
    def __len__(self):
        return len(self.dataset)
    
    def save_cache_file(self, path, data):
        with open(path, "wb") as fp:
            pickle.dump(data, fp)
    
    def load_cache_file(self, path):
        with open(path, "rb") as fp:
            return pickle.load(fp)
    
    @abstractmethod
    def build_dataset(self, data):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
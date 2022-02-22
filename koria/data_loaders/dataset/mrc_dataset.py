from asyncio.log import logger
import os
import pickle
import torch
import logging

logger = logging.getLogger("koria")

from collections import deque
from tqdm import tqdm
from torch.utils.data import Dataset


class MrcDataset(Dataset):
    
    def __init__(self, tokenizer, model_name, data_list, cache_dir, vocab_dir, dataset_type="train", max_seq_len=256, special_tokens=None):
        super().__init__()
        
        # Definition of special tokens
        self.LABEL_PAD_TOKEN = "<PAD>"
        
        self.SPECIAL_LABEL_TOKENS = {
            "pad": self.LABEL_PAD_TOKEN
        }
    
        # add predicate tokens into tokenizer
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        if self.model_name == "bert":
            self.special_tokenizer_sep_indicator = "▁"
            self.special_tokenizer_replaced_sep_indicator = " "
        else:
            self.special_tokenizer_sep_indicator = "##"
            self.special_tokenizer_replaced_sep_indicator = ""
            
        self.max_seq_len = max_seq_len
        self.context_tokens = []
        self.question_tokens = []
        self.answers = []
        
        cache_path = os.path.join(cache_dir, f"mrc-{dataset_type}-{model_name}-data.cache")
        
        # We use cache file for improving data loading speed when cache file is existed in cache directory.
        # But file is not existed, then build dataset and save into cache file
        if not os.path.isfile(cache_path):
            
            for data in tqdm(data_list, desc=f"Load {dataset_type} data"):
                context = data["context"]
                question = data["question"]
                answer = data["answer"]

                try:
                    context_token_list = tokenizer.tokenize(context)
                    question_token_list = tokenizer.tokenize(question)
                    
                    answer = self.adjust_answer_position(context_tokens=context_token_list, question_tokens=question_token_list, answer=answer)
                    
                    self.context_tokens.append(context_token_list)
                    self.question_tokens.append(question_token_list)
                    self.answers.append(answer)
                    
                except IndexError:
                    pass
            
            # Save cache file
            with open(cache_path, "wb") as fp:
                pickle.dump({
                    "context_tokens": self.context_tokens, 
                    "question_tokens": self.question_tokens,
                    "answers": self.answers
                }, fp)
                    
        else:
            # Load cache and vocab files
            with open(cache_path, "rb") as fp:
                data = pickle.load(fp)
                
            if "tokens" not in data.keys() or "labels" not in data.keys():
                raise KeyError("Invalid cache file. Please cache file and run it again")
            
            self.tokens = data["tokens"]
            self.labels = data["labels"]
        
    def adjust_answer_position(self, context_tokens, question_tokens, answer):
        return answer
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        token_list = [self.tokenizer.cls_token] + self.tokens[idx]
        label_list = ["O"] + self.labels[idx]
        
        if len(token_list) <= self.max_seq_len:
            token_list += [self.tokenizer.pad_token] * (self.max_seq_len - len(token_list))
            label_list += [self.LABEL_PAD_TOKEN] * (self.max_seq_len - len(label_list))
        else:
            token_list = token_list[:self.max_seq_len]
            label_list = label_list[:self.max_seq_len]
            
        token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        token_type_ids = [1] * len(token_list)
        label_ids = [self.l2i[label] for label in label_list]
        
        
        return torch.tensor(token_ids), torch.tensor(token_type_ids), torch.tensor(label_ids)
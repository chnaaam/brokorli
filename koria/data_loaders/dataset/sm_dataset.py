import torch
import logging

logger = logging.getLogger("koria")

from . import DatasetBase


class SmDataset(DatasetBase):
    
    def __init__(
        self, 
        tokenizer, 
        task_name, 
        model_name, 
        data_list, 
        cache_dir, 
        vocab_dir, 
        dataset_type="train", 
        max_seq_len=256, 
        special_tokens=None):
        
        super().__init__(
            tokenizer=tokenizer,
            task_name=task_name,
            model_name=model_name,
            data_list=data_list,
            cache_dir=cache_dir,
            vocab_dir=vocab_dir,
            dataset_type=dataset_type,
            max_seq_len=max_seq_len,
            special_tokens=special_tokens,
            build_dataset_func=self.build_dataset
        )
        
    def build_dataset(self, data):
        context = data["context"]
        question = data["question"]
        label = data["label"]

        try:
            context_token_list = self.tokenizer.tokenize(context)
            question_token_list = self.tokenizer.tokenize(question)
            
            return {
                "context": context_token_list,
                "question": question_token_list,
                "label": label
            }
            
        except IndexError:
            return None
    
    def __getitem__(self, idx):
        context_tokens = self.dataset[idx]["context"]
        question_tokens = self.dataset[idx]["question"]
        label = self.dataset[idx]["label"]
        
        token_list = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token] + context_tokens
        token_type_ids = [0] * len([self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]) + [1] * len(context_tokens)
        
        if len(token_list) > self.max_seq_len:
            token_list = token_list[:self.max_seq_len]
            token_type_ids = token_type_ids[:self.max_seq_len]
        else:
            token_list += [self.tokenizer.pad_token] * (self.max_seq_len - len(token_list))
            token_type_ids += [1] * (self.max_seq_len - len(token_type_ids))
        
        token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        
        
        return torch.tensor(token_ids), torch.tensor(token_type_ids), torch.tensor(label)
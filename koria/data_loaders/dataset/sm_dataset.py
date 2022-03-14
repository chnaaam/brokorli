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
        label_dir, 
        dataset_type="train", 
        max_seq_len=256):
        
        super().__init__(
            tokenizer=tokenizer,
            task_name=task_name,
            model_name=model_name,
            data_list=data_list,
            cache_dir=cache_dir,
            label_dir=label_dir,
            dataset_type=dataset_type,
            max_seq_len=max_seq_len,
            build_dataset_func=self.build_dataset
        )
        
        self.build_vocab(vocab=[True, False])
        
    def build_dataset(self, data):
        context = data["sentence"]
        question = data["question"]
        label = data["label"]

        try:
            context_token_list = self.tokenizer.tokenize(context)
            question_token_list = self.tokenizer.tokenize(question)
            
            return {
                "sentence": context_token_list,
                "question": question_token_list,
                "label": label
            }
            
        except IndexError:
            return None
    
    def __getitem__(self, idx):
        context_tokens = self.dataset[idx]["sentence"]
        question_tokens = self.dataset[idx]["question"]
        label = self.dataset[idx]["label"]
        
        token_list = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token] + context_tokens + [self.tokenizer.sep_token]
        token_type_ids = [0] * len([self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]) + [1] * (len(context_tokens) + 1)
        
        if len(token_list) > self.max_seq_len:
            token_list = token_list[:self.max_seq_len]
            token_type_ids = token_type_ids[:self.max_seq_len]
        else:
            token_list += [self.tokenizer.pad_token] * (self.max_seq_len - len(token_list))
            token_type_ids += [1] * (self.max_seq_len - len(token_type_ids))
        
        token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        label_ids = self.l2i[label]
        
        input_ids = torch.tensor(token_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        label_ids = torch.tensor(label_ids)
        
        return input_ids, token_type_ids, attention_mask, label_ids
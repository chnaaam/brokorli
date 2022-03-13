import torch
import logging

logger = logging.getLogger("koria")

from collections import deque

from . import DatasetBase
from koria.special_tokens import LABEL_PAD_TOKEN
class NerDataset(DatasetBase):
    
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
        
        vocab = [LABEL_PAD_TOKEN]
        for dataset in self.dataset:
            vocab += dataset["labels"]
        
        self.build_vocab(vocab=list(set(vocab)))
    
    def build_dataset(self, data):
        sentence = data["sentence"]
        entities = data["entities"]
        
        if not entities:
            return None
        
        """
        entities example
        [
            {"id": 1, "form": "멕시코", "label": "LC", "begin": 0, "end": 3}, 
            {"id": 2, "form": "파키스탄", "label": "LC", "begin": 4, "end": 8}, 
            {"id": 3, "form": "방글라데시", "label": "LC", "begin": 9, "end": 14}, 
            {"id": 4, "form": "베트남", "label": "LC", "begin": 15, "end": 18}, 
            {"id": 5, "form": "근로자", "label": "CV", "begin": 27, "end": 30}
        ]
        """
        
        try:
            token_list = self.tokenizer.tokenize(sentence)
            label_list = []

            label_list = self.adjust_label_position(sentence, len(token_list), entities)
            label_list = self.convert_plain_label_to_bioes_tag(label_list=label_list)
                        
            return {
                "tokens": token_list,
                "labels": label_list
            }
            
        except IndexError:
            return None
    
    
    def adjust_label_position(self, sentence, len_token_list, entities):
        offsets = self.tokenizer(sentence, return_offsets_mapping=True)["offset_mapping"]
        entities = deque(entities)
        
        label_list = ["O"] * len_token_list
        
        for offset_idx, (word_begin_idx, word_end_idx) in enumerate(offsets):
            
            if offset_idx == 0:
                continue
            
            word_end_idx -= 1
            
            if word_begin_idx > entities[0]["end"] - 1:
                entities.popleft()
            
            if not entities:
                break
            
            begin_idx, end_idx = entities[0]["begin"], entities[0]["end"] - 1
            
            if word_begin_idx <= begin_idx <= word_end_idx or \
                word_begin_idx <= end_idx <= word_end_idx or \
                begin_idx <= word_begin_idx <= end_idx or \
                begin_idx <= word_end_idx <= end_idx:
                    
                label_list[offset_idx - 1] = entities[0]["label"]
                
        return label_list
                
    def convert_plain_label_to_bioes_tag(self, label_list):
        previous_label = "O"
        for idx in range(len(label_list)):
            label = label_list[idx]
            
            if label == "O":
                previous_label = "O"
                continue
            
            if previous_label != label:
                if idx != len(label_list) - 1:
                    if label == label_list[idx + 1]:
                        label_list[idx] = "B-" + label
                    else:
                        label_list[idx] = "S-" + label
                else:
                    label_list[idx] = "S-" + label
                    
            else:
                if idx == len(label_list) - 1:
                    label_list[idx] = "E-" + label     
                else:
                    if label == label_list[idx + 1]:
                        label_list[idx] = "I-" + label
                    else:
                        label_list[idx] = "E-" + label
            
            previous_label = label
            
        return label_list
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        tokens = self.dataset[idx]["tokens"]
        labels = self.dataset[idx]["labels"]
        
        token_list = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        token_type_ids = [1] * len(token_list)
        label_list = ["O"] + labels + ["O"]
        
        if len(token_list) <= self.max_seq_len:
            token_list += [self.tokenizer.pad_token] * (self.max_seq_len - len(token_list))
            token_type_ids += [0] * (self.max_seq_len - len(token_type_ids))
            label_list += [LABEL_PAD_TOKEN] * (self.max_seq_len - len(label_list))
        else:
            token_list = token_list[:self.max_seq_len]
            token_type_ids = token_type_ids[:self.max_seq_len]
            label_list = label_list[:self.max_seq_len]
            
            if label_list[-1] != "O":
                if label_list[-1].startswith("I-"):
                    label_list[-1] = label_list[-1].replace("I-", "E-")
                elif label_list[-1].startswith("B-"):
                    label_list[-1] = label_list[-1].replace("B-", "S-")
                
            
        token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        label_ids = [self.l2i[label] for label in label_list]
        
        input_ids = torch.tensor(token_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        label_ids = torch.tensor(label_ids)
        
        return input_ids, token_type_ids, attention_mask, label_ids
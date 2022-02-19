import os
import pickle
import torch

from collections import deque
from tqdm import tqdm
from torch.utils.data import Dataset

class SrlDataset(Dataset):
    
    def __init__(self, tokenizer, special_tokens, model_name, data_list, cache_dir, vocab_dir, dataset_type="train", max_seq_len=256):
        super().__init__()
        
        # Definition of special tokens   
        self.LABEL_PAD_TOKEN = "<PAD>"
        self.START_OF_PREDICATE_SPECIAL_TOKEN = special_tokens["predicate_begin"]
        self.END_OF_PREDICATE_SPECIAL_TOKEN = special_tokens["predicate_end"]
        
        self.SPECIAL_LABEL_TOKENS = {
            "pad": self.LABEL_PAD_TOKEN
        }
    
        # add predicate tokens into tokenizer
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.tokens = []
        self.labels = []
        
        cache_path = os.path.join(cache_dir, f"srl-{dataset_type}-{model_name}-data.cache")
        
        # we use cache file for improving data loading speed when cache file is existed in cache directory.
        # But file is not existed, then build dataset and save into cache file
        count = 0
        if not os.path.isfile(cache_path):
            
            for data in tqdm(data_list, desc=f"Tokenize {dataset_type} data"):
                sentence = data["sentence"]
                predicate = data["predicate"]
                arguments = data["arguments"]
                
                if not arguments:
                    continue
                
                # Check nested proposition
                is_passed = False
                for argument in arguments:
                    if argument["begin"] == predicate["begin"] or argument["end"] == predicate["end"]:
                        is_passed = True
                
                if is_passed:
                    continue
                
                sentence.replace("\\\"", "\"")
                
                """
                predicate example
                => {"form": "분양하는", "begin": 18, "end": 22, "lemma": "분양하다"}
                
                argument example
                [
                    {"form": "경기 성남시 판교신도시", "label": "ARGM-LOC", "begin": 0, "end": 12, "word_id": 3},
                    {"form": "이달", "label": "ARGM-TMP", "begin": 15, "end": 17, "word_id": 4}
                ]
                """
                
                try:
                    predicate_begin_idx, predicate_end_idx = predicate["begin"], predicate["end"]
                    
                    original_sentence = sentence
                    sentence = sentence[:predicate_begin_idx] + self.START_OF_PREDICATE_SPECIAL_TOKEN + sentence[predicate_begin_idx: predicate_end_idx] + self.END_OF_PREDICATE_SPECIAL_TOKEN + sentence[predicate_end_idx:]
                    token_list = tokenizer.tokenize(sentence)
                    label_list = []

                    for idx in range(len(arguments)):
                        arg_end_idx = int(arguments[idx]["end"])
                        
                        if arg_end_idx > predicate_end_idx:
                            arguments[idx]["begin"] += 2
                            arguments[idx]["end"] += 2
                    
                    char_label_list =  self.convert_word_pos_to_char_pos(sentence=original_sentence, arguments=arguments)
                    label_list = self.convert_char_pos_to_token_pos(token_list=token_list, char_label_list=char_label_list)                    
                    label_list = self.convert_plain_label_to_bioes_tag(label_list=label_list)
                    
                    self.tokens.append(token_list)
                    self.labels.append(label_list)
                        
                except IndexError:
                    count += 1
            
            # Save cache file
            with open(cache_path, "wb") as fp:
                pickle.dump({"tokens": self.tokens, "labels": self.labels}, fp)
                    
        else:
            # Load cache and vocab files
            with open(cache_path, "rb") as fp:
                data = pickle.load(fp)
                
            if "tokens" not in data.keys() or "labels" not in data.keys():
                raise KeyError("Invalid cache file. Please cache file and run it again")
            
            self.tokens = data["tokens"]
            self.labels = data["labels"]
        
        # Load or build labels
        vocab_path = os.path.join(vocab_dir, "srl.label")
        
        if not os.path.isfile(vocab_path):
            vocab = []    
            for labels in self.labels:
                vocab += labels
                
            self.vocab = list(set(vocab))   
            self.vocab = list(self.SPECIAL_LABEL_TOKENS.values()) + self.vocab
            
            self.l2i = {l: i for i, l in enumerate(self.vocab)}
            self.i2l = {i: l for l, i in self.l2i.items()}
        
            with open(vocab_path, "wb") as fp:
                pickle.dump({"l2i": self.l2i}, fp)
        
        else:
            with open(vocab_path, "rb") as fp:
                data = pickle.load(fp)

            if "l2i" not in data.keys():
                raise KeyError("Invalid label file. Please label file and run it again")

            self.l2i = data["l2i"]
            self.vocab = list(set(self.l2i.keys()))
            self.i2l = {i: l for l, i in self.l2i.items()}
            
    def convert_word_pos_to_char_pos(self, sentence, arguments):
        # +2 - Predicate Tokens 
        char_label_list = ["O" for _ in range(len(sentence) + 2)]

        for argument in arguments:
            begin_idx = int(argument["begin"])
            end_idx = int(argument["end"])
            
            char_label_list[begin_idx: end_idx] = [argument["label"]] * (end_idx - begin_idx)
            
        return char_label_list
    
    def convert_char_pos_to_token_pos(self, token_list, char_label_list):
        label_list = []
        char_label_list = deque(char_label_list)

        for token_idx, token in enumerate(token_list):
            label = "O"
            
            if token_idx == 0:
                token = token[1:]
            else:
                if token.startswith("▁"):
                    token = token.replace("▁", " ")
                
            if not token:
                token = " "
            
            if token == self.START_OF_PREDICATE_SPECIAL_TOKEN or token == self.END_OF_PREDICATE_SPECIAL_TOKEN:
                char_label_list.popleft()
                label_list.append(label)
                continue
            
            while token:
                char_label = char_label_list.popleft()
                
                token = token[1:]
                
                if char_label != "O":
                    label = char_label
            
            label_list.append(label)
            
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
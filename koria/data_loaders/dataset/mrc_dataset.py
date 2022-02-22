import torch
import logging

logger = logging.getLogger("koria")

from . import DatasetBase


class MrcDataset(DatasetBase):
    
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
        answer = data["answer"]

        try:
            context_token_list = self.tokenizer.tokenize(context)
            question_token_list = self.tokenizer.tokenize(question)
            
            answer = self.adjust_answer_position(context_tokens=context_token_list, answer=answer)
            
            return {
                "context": context_token_list,
                "question": question_token_list,
                "answer": answer
            }
            
        except IndexError:
            return None
    
    def adjust_answer_position(self, context_tokens, answer):
        # BERT
        token_start_idx = 0
        token_end_idx = 0
        
        answer_begin_idx = answer["begin"]
        answer_end_idx = answer["end"]
        adjusted_answer = {
            "begin": -1,
            "end": -1
        }
        
        for token_idx, context_token in enumerate(context_tokens):
            
            if context_token.startswith("▁"):
                context_token.replace("▁", " ")
            
            if token_idx == 0:
                context_token = context_token[1:]
                token_start_idx = token_end_idx
            else:
                token_start_idx = token_end_idx + 1
            token_end_idx = token_start_idx + len(context_token) - 1
            
            
            if token_start_idx <= answer_begin_idx <= token_end_idx \
                and token_start_idx <= answer_end_idx <= token_end_idx:
                    adjusted_answer["begin"] = token_idx
                    adjusted_answer["end"] = token_idx
            
            elif answer_begin_idx <= token_start_idx <= answer_end_idx \
                or answer_begin_idx <= token_start_idx <= answer_end_idx \
                or token_start_idx <= answer_begin_idx <= token_end_idx \
                or token_start_idx <= answer_end_idx <= token_end_idx:
                    
                if adjusted_answer["begin"] == -1:
                    adjusted_answer["begin"] = token_idx
                else:
                    adjusted_answer["end"] = token_idx
            
        
        return answer
    
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
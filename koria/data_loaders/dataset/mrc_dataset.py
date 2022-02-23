from tkinter import E
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
            
        
        return adjusted_answer
    
    def __getitem__(self, idx):
        context_tokens = self.dataset[idx]["context"]
        question_tokens = self.dataset[idx]["question"]
        
        answer_begin_idx = self.dataset[idx]["answer"]["begin"]
        answer_end_idx = self.dataset[idx]["answer"]["end"]
        
        len_c_tokens = len(context_tokens)
        len_q_tokens = len(question_tokens)
        
        # huggingface run_squad.py idea
        doc_stride = 64
        
        adjusted_len_c_tokens = self.max_seq_len - 2 - len_q_tokens
        
        if adjusted_len_c_tokens < len_c_tokens:
            
            begin_doc_stride = doc_stride
            end_doc_stride = doc_stride
            
            if answer_begin_idx - begin_doc_stride < 0:
                begin_doc_stride = 0
            
            if answer_end_idx + end_doc_stride > len_c_tokens:
                # end_doc_stride = answer_end_idx + end_doc_stride - len_c_tokens
                end_doc_stride = len_c_tokens - answer_end_idx
                
            if answer_end_idx + end_doc_stride > adjusted_len_c_tokens:
                # TODO ...
                
                # Context token slide window
                ct_end_idx = answer_end_idx + end_doc_stride
                ct_begin_idx = ct_end_idx - adjusted_len_c_tokens
                
                
                # Adjust answer begin and end index
                answer_be_interval = answer_end_idx - answer_begin_idx
                
                answer_end_idx = adjusted_len_c_tokens - end_doc_stride
                answer_begin_idx = answer_end_idx - answer_be_interval
                
            else:
                ct_begin_idx = 0
                ct_end_idx = adjusted_len_c_tokens
                
            context_tokens = context_tokens[ct_begin_idx: ct_end_idx]
        else:
            context_tokens += [self.tokenizer.pad_token] * (adjusted_len_c_tokens - len_c_tokens)
            
        answer_begin_idx += len_q_tokens + 2
        answer_end_idx += len_q_tokens + 2
        
        token_list = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token] + context_tokens
        
        token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        token_type_ids = [0] * len([self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]) + [1] * len(context_tokens)
        
        return torch.tensor(token_ids), torch.tensor(token_type_ids), torch.tensor(answer_begin_idx), torch.tensor(answer_end_idx)
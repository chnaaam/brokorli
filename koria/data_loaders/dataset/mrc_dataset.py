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
        
    def build_dataset(self, data):
        context = data["context"]
        question = data["question"]
        answer = data["answer"]

        try:
            if context[data["answer"]["begin"]: data["answer"]["end"]+1] == "1839":
                print()
                
            context_token_list = self.tokenizer.tokenize(context)
            question_token_list = self.tokenizer.tokenize(question)
            
            answer = self.adjust_answer_position(context=context, answer=answer)
            
            if self.tokenizer.convert_tokens_to_string(context_token_list[answer["begin"]: answer["end"]+1]) != context[data["answer"]["begin"]: data["answer"]["end"]+1]:
                print()
            return {
                "context": context_token_list,
                "question": question_token_list,
                "answer": answer
            }
            
        except IndexError:
            return None
    
    def adjust_answer_position(self, context, answer):
        # TODO: Unsupported KoBERT and KoCharElectra Tokenizer
        
        answer_begin_idx = answer["begin"]
        answer_end_idx = answer["end"]
        adjusted_answer = {
            "begin": -1,
            "end": -1
        }
        
        offsets = self.tokenizer(context, return_offsets_mapping=True)["offset_mapping"][:-1]
        
        for offset_idx, (word_begin_idx, word_end_idx) in enumerate(offsets):
            
            if offset_idx == 0:
                continue
            
            word_end_idx = word_end_idx - 1
            
            if word_begin_idx <= answer_begin_idx <= word_end_idx and word_begin_idx <= answer_end_idx <= word_end_idx:
                adjusted_answer["begin"] = offset_idx - 1
                adjusted_answer["end"] = offset_idx - 1
                
                break
            
            if word_begin_idx <= answer_begin_idx <= word_end_idx or \
                word_begin_idx <= answer_end_idx <= word_end_idx or \
                answer_begin_idx <= word_begin_idx <= answer_end_idx or \
                answer_begin_idx <= word_end_idx <= answer_end_idx:
                
                if adjusted_answer["begin"] == -1:
                    adjusted_answer["begin"] = offset_idx - 1   # CLS Token - 1
                else:
                    adjusted_answer["end"] = offset_idx - 1
            
        return adjusted_answer
    
    def __getitem__(self, idx):
        context_tokens = self.dataset[idx]["context"]
        question_tokens = self.dataset[idx]["question"]
        
        answer_begin_idx = self.dataset[idx]["answer"]["begin"]
        answer_end_idx = self.dataset[idx]["answer"]["end"]
        
        len_c_tokens = len(context_tokens)
        len_q_tokens = len(question_tokens)
        
        # huggingface run_squad.py
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
        
        input_ids = torch.tensor(token_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        answer_begin_idx_tensor = torch.tensor(answer_begin_idx)
        answer_end_idx_tensor = torch.tensor(answer_end_idx)
        
        return input_ids, token_type_ids, attention_mask, answer_begin_idx_tensor, answer_end_idx_tensor
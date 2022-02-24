import os
import torch
from tqdm import tqdm

from .task_base import TaskBase

from koria.metrics import calculate_qa_f1_score, calculate_em_score

class MRC(TaskBase):

    def __init__(self, **parameters):
                
        # Set common parameters for TaskBase and build model
        super().__init__(model_parameters=None, **parameters)
        
    def train(self):
        
        max_score = 0
        
        for epoch in range(int(self.cfg.epochs)):
            self.model.train()
            train_losses = []
            avg_train_loss = 0
            
            progress_bar = tqdm(self.train_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Training] Epoch : {epoch}, Avg Loss : {avg_train_loss:.4f}")
                
                token_tensor, token_type_ids_tensor, answer_begin_idx_tensor, answer_end_idx_tensor = data
                
                token_tensor.to(self.device)
                token_type_ids_tensor.to(self.device)
                answer_begin_idx_tensor.to(self.device)
                answer_end_idx_tensor.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=(token_tensor != self.token_pad_id).float(),
                    start_positions=answer_begin_idx_tensor,
                    end_positions=answer_end_idx_tensor,
                )
                
                loss = outputs["loss"]
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
            
            avg_valid_loss, avg_valid_f1_score, avg_valid_em_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss:.4f}\tValid Loss : {avg_valid_loss:.4f}\tValid F1 Score : {avg_valid_f1_score * 100:.4f}\tEM Score : {avg_valid_em_score * 100:.4f}")
            
            if max_score < avg_valid_f1_score:
                self.save_model(path=os.path.join(self.model_hub_path, f"mrc-e{epoch}-f1{avg_valid_f1_score * 100:.4f}-em{avg_valid_em_score * 100:.4f}-lr{self.cfg.learning_rate}-len{self.max_seq_len}.mdl"))
                
    def valid(self):
        self.model.eval()
        
        with torch.no_grad():
            valid_losses, valid_f1_scores, valid_em_scores = [], [], []
            avg_valid_loss, avg_valid_f1_score, avg_valid_em_score = 0, 0, 0
            
            progress_bar = tqdm(self.valid_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Validation] Avg Loss : {avg_valid_loss:.4f} Avg Score : {avg_valid_f1_score * 100:.4f}")
                
                token_tensor, token_type_ids_tensor, answer_begin_idx_tensor, answer_end_idx_tensor = data
                attention_mask = (token_tensor != self.token_pad_id).float()
                
                self.model.to(self.device)
                
                token_tensor = token_tensor.to(self.device)
                token_type_ids_tensor = token_type_ids_tensor.to(self.device)
                attention_mask = attention_mask.to(self.device)
                answer_begin_idx_tensor = answer_begin_idx_tensor.to(self.device)
                answer_end_idx_tensor = answer_end_idx_tensor.to(self.device)
                                
                outputs = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=attention_mask,
                    start_positions=answer_begin_idx_tensor,
                    end_positions=answer_end_idx_tensor,
                )
                
                loss = outputs["loss"]
                start_logits = outputs["start_logits"]
                end_logits = outputs["end_logits"]
                
                pred_begin_indexes = torch.argmax(start_logits, dim=-1)
                pred_end_indexes = torch.argmax(end_logits, dim=-1)
                
                valid_losses.append(loss.item())
                
                true_answers, pred_answers = self.decode(
                    token_tensor,
                    answer_begin_idx_tensor.tolist(), 
                    answer_end_idx_tensor.tolist(), 
                    pred_begin_indexes.tolist(), 
                    pred_end_indexes.tolist())
                
                f1_score = calculate_qa_f1_score(true_answers, pred_answers)
                em_score = calculate_em_score(true_answers, pred_answers)
                        
                valid_f1_scores.append(f1_score)
                valid_em_scores.append(em_score)
                
                avg_valid_loss = sum(valid_losses) / len(valid_losses)    
                avg_valid_f1_score = sum(valid_f1_scores) / len(valid_f1_scores)    
                avg_valid_em_score = sum(valid_em_scores) / len(valid_em_scores)    
                
            return avg_valid_loss, avg_valid_f1_score, avg_valid_em_score
    
    def test(self):
        pass
    
    def predict(self, sentence, question):
        self.load_model(path=os.path.join(self.model_hub_path, "mrc.mdl"))
        self.model.eval()
        
        with torch.no_grad():
            sentence_tokens = self.tokenizer.tokenize(sentence)
            question_tokens = self.tokenizer.tokenize(question)
            
            token_list = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]
            len_question_token_list = len(token_list)
            
            if len(sentence_tokens) > self.max_seq_len - len_question_token_list:
                sentence_tokens = sentence_tokens[:self.max_seq_len - len_question_token_list]
            else:
                sentence_tokens = sentence_tokens + [self.tokenizer.pad_token] * (self.max_seq_len - len_question_token_list - len(sentence_tokens))
            
            token_list = token_list + sentence_tokens
            
            token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
            token_type_ids = [0] * len([self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]) + [1] * len(sentence_tokens)
            
            token_tensor, token_type_ids_tensor = torch.tensor([token_ids]), torch.tensor([token_type_ids])
            attention_mask = (token_tensor != self.tokenizer.pad_token_id).float()
            
            self.model.to(self.device)
            
            token_tensor = token_tensor.to(self.device)
            token_type_ids_tensor = token_type_ids_tensor.to(self.device)
            attention_mask = attention_mask.to(self.device)
                            
            outputs = self.model(
                token_tensor, 
                token_type_ids=token_type_ids_tensor,
                attention_mask=attention_mask,
                start_positions=None,
                end_positions=None,
            )
            
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            
            pred_begin_indexes = torch.argmax(start_logits, dim=-1)
            pred_end_indexes = torch.argmax(end_logits, dim=-1)
            
            answer = self.tokenizer.convert_tokens_to_string(token_list[pred_begin_indexes: pred_end_indexes + 1])
            
            print(answer)
            return answer
        
    def decode(self, token_tensor, true_begin_indexes, true_end_indexes, pred_begin_indexes, pred_end_indexes):
        true_answers, pred_answers = [], []
        
        for idx, tt in enumerate(token_tensor):
            tokens = self.tokenizer.convert_ids_to_tokens(tt)
            
            true_begin_idx, true_end_idx = true_begin_indexes[idx], true_end_indexes[idx]
            pred_begin_idx, pred_end_idx = pred_begin_indexes[idx], pred_end_indexes[idx]
            
            true_answer = tokens[true_begin_idx: true_end_idx + 1]
            
            if pred_begin_idx <= pred_end_idx:
                pred_answer = tokens[pred_begin_idx: pred_end_idx + 1]
            else:
                pred_answer = []
                
            true_answers.append(self.tokenizer.convert_tokens_to_string(true_answer))
            pred_answers.append(self.tokenizer.convert_tokens_to_string(pred_answer))
            
        return true_answers, pred_answers
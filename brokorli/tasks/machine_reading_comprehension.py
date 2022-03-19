import torch
import torch.nn.functional as F
from tqdm import tqdm

from .neural_based_task import NeuralBaseTask

from brokorli.metrics import calculate_qa_f1_score, calculate_em_score

class MRC(NeuralBaseTask):

    def __init__(self, task_config):
        super().__init__(config=task_config)
        
    def train(self):
        max_score = 0
        
        for epoch in range(int(self.config.epochs)):
            self.model.train()
            
            train_losses = []
            avg_train_loss = 0
            
            progress_bar = tqdm(self.train_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Training] Epoch : {epoch}, Avg Loss : {avg_train_loss:.4f}")
                
                input_ids, token_type_ids, attention_mask, answer_begin_idx_tensor, answer_end_idx_tensor = data
                
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    start_positions=answer_begin_idx_tensor,
                    end_positions=answer_end_idx_tensor,
                )
                
                loss = outputs["loss"]
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
                
            avg_valid_loss, avg_valid_f1_score, avg_valid_em_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss:.4f}\tValid Loss : {avg_valid_loss:.4f}\tValid F1 Score : {avg_valid_f1_score * 100:.4f}\tValid F1 Score : {avg_valid_em_score * 100:.4f}")
            
            if max_score < avg_valid_f1_score:
                self.update_trained_model(self.MODEL_PATH.format(epoch, avg_valid_f1_score * 100))
                max_score = avg_valid_f1_score
                
    def valid(self):
        self.model.eval()
        
        with torch.no_grad():
            valid_losses, valid_f1_scores, valid_em_scores = [], [], []
            avg_valid_loss, avg_valid_f1_score, avg_valid_em_score = 0, 0, 0
            
            progress_bar = tqdm(self.config.valid_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Validation] Avg Loss : {avg_valid_loss:.4f} Avg Score : {avg_valid_f1_score * 100:.4f}")
                
                input_ids, token_type_ids, attention_mask, answer_begin_idx_tensor, answer_end_idx_tensor = data
                input_ids, token_type_ids, attention_mask, answer_begin_idx_tensor, answer_end_idx_tensor = (
                    input_ids.to(self.device),
                    token_type_ids.to(self.device),
                    attention_mask.to(self.device),
                    answer_begin_idx_tensor.to(self.device),
                    answer_end_idx_tensor.to(self.device)
                )
                                
                outputs = self.model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids,
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
                    input_ids,
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
    
    def predict(self, **parameters):
        
        if "sentence" not in parameters.keys() or "question" not in parameters.keys():
            raise KeyError("The machine reading comprehension task must need sentence and question parameters")
        
        sentence = parameters["sentence"]
        question = parameters["question"]
        
        with torch.no_grad():
            if type(question) == str:
                question = [question]
                
            if type(sentence) == str:
                sentence = [sentence] * len(question)
                
            tokenized = self.tokenizer(question, sentence, padding="max_length", max_length=self.config.max_seq_len, truncation=True, return_tensors="pt")
            outputs = self.model(
                input_ids=tokenized["input_ids"].to(self.device), 
                token_type_ids=tokenized["token_type_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device),
                start_positions=None,
                end_positions=None,
            )
            
            pred_begin_scores = F.softmax(outputs["start_logits"], dim=-1)
            pred_end_scores = F.softmax(outputs["end_logits"], dim=-1)
            
            answers = []
            for idx, (pred_begin_index, pred_end_index) in enumerate(zip(torch.argmax(outputs["start_logits"], dim=-1), torch.argmax(outputs["end_logits"], dim=-1))):                
                answers.append({
                    "answer": self.tokenizer.decode(tokenized["input_ids"][idx][pred_begin_index: pred_end_index + 1]),
                    "confidence_score": (pred_begin_scores[idx][pred_begin_index] + pred_end_scores[idx][pred_end_index]).item() / 2
                })
            
            return answers
        
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
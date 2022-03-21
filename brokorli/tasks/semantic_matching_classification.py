import torch
import torch.functional as F
from tqdm import tqdm

from .neural_based_task import NeuralBaseTask

from brokorli.metrics import calculate_sc_score

class SM(NeuralBaseTask):

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
                
                input_ids, token_type_ids, attention_mask, label_ids = data
                
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=label_ids,
                )
                
                loss = outputs["loss"]
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
            
            avg_valid_loss, avg_valid_f1_score, avg_valid_acc_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss:.4f}\tValid Loss : {avg_valid_loss:.4f}\tValid F1 Score : {avg_valid_f1_score * 100:.4f}\tAcc Score : {avg_valid_acc_score * 100:.4f}")
            
            if max_score < avg_valid_f1_score:
                self.update_trained_model(self.MODEL_PATH.format(epoch, avg_valid_f1_score * 100))
                max_score = avg_valid_f1_score
                
    def valid(self):
        self.model.eval()
        
        with torch.no_grad():
            valid_losses, valid_f1_scores, valid_acc_scores = [], [], []
            avg_valid_loss, avg_valid_f1_score, avg_valid_acc_score = 0, 0, 0
            
            progress_bar = tqdm(self.config.test_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Validation] Avg Loss : {avg_valid_loss:.4f} Avg Score : {avg_valid_f1_score * 100:.4f}")
                
                input_ids, token_type_ids, attention_mask, label_ids = data
                
                input_ids, token_type_ids, attention_mask, label_ids = (
                    input_ids.to(self.device),
                    token_type_ids.to(self.device),
                    attention_mask.to(self.device),
                    label_ids.to(self.device)
                )
                
                outputs = self.model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=label_ids,
                )
                
                loss = outputs["loss"]
                logits = outputs["logits"]                
                
                valid_losses.append(loss.item())
                
                pred_tags = torch.argmax(logits, dim=-1)
                score = calculate_sc_score(true_y=label_ids.float().tolist(), pred_y=pred_tags.tolist())
                
                
                valid_f1_scores.append(score["f1"])
                valid_acc_scores.append(score["accuracy"])
                
                avg_valid_loss = sum(valid_losses) / len(valid_losses)    
                avg_valid_f1_score = sum(valid_f1_scores) / len(valid_f1_scores)    
                avg_valid_acc_score = sum(valid_acc_scores) / len(valid_acc_scores)    
                
            return avg_valid_loss, avg_valid_f1_score, avg_valid_acc_score
        
    def predict(self, **parameters):
        
        if "sentence" not in parameters.keys() or "question" not in parameters.keys():
            raise KeyError("The machine reading comprehension task must need sentence and question parameters")
        
        # TODO : sentence length = 1 and question length > 1        
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
                labels=None,
            )
            
            logits = outputs["logits"]
            
            pred_label = torch.argmax(logits, dim=-1).tolist()
            
            return [self.i2l[label] for label in pred_label]
        
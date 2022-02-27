import os
import torch
from tqdm import tqdm

from .neural_based_task import NeuralBaseTask

from koria.metrics import calculate_sc_score

class SM(NeuralBaseTask):

    def __init__(self, **parameters):
                
        # Set common parameters for TaskBase and build model
        super().__init__(model_parameters=None, **parameters)
        
    def train(self):
        max_score = 0
        
        for epoch in range(int(self.cfg.parameters.epochs)):
            self.model.train()
            train_losses = []
            avg_train_loss = 0
            
            progress_bar = tqdm(self.train_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Training] Epoch : {epoch}, Avg Loss : {avg_train_loss:.4f}")
                
                token_tensor, token_type_ids_tensor, label_tensor = data
                
                token_tensor.to(self.device)
                token_type_ids_tensor.to(self.device)
                label_tensor.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=(token_tensor != self.tokenizer.pad_token_id).float(),
                    labels=label_tensor,
                )
                
                loss = outputs["loss"]
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
            
            avg_valid_loss, avg_valid_f1_score, avg_valid_acc_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss:.4f}\tValid Loss : {avg_valid_loss:.4f}\tValid F1 Score : {avg_valid_f1_score * 100:.4f}\tEM Score : {avg_valid_acc_score * 100:.4f}")
            
            if max_score < avg_valid_f1_score:
                self.save_model(path=os.path.join(self.model_hub_path, f"mrc-e{epoch}-f1{avg_valid_f1_score * 100:.4f}-acc{avg_valid_acc_score * 100:.4f}-lr{self.cfg.learning_rate}-len{self.max_seq_len}.mdl"))
                
    def valid(self):
        self.model.eval()
        
        with torch.no_grad():
            valid_losses, valid_f1_scores, valid_acc_scores = [], [], []
            avg_valid_loss, avg_valid_f1_score, avg_valid_acc_score = 0, 0, 0
            
            progress_bar = tqdm(self.valid_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Validation] Avg Loss : {avg_valid_loss:.4f} Avg Score : {avg_valid_f1_score * 100:.4f}")
                
                token_tensor, token_type_ids_tensor, label_tensor = data
                if self.use_cuda:
                    token_tensor, token_type_ids_tensor, label_tensor = token_tensor.cuda(), token_type_ids_tensor.cuda(), label_tensor.cuda()
                
                outputs = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=(token_tensor != self.tokenizer.pad_token_id).float(),
                    labels=label_tensor,
                )
                
                loss = outputs["loss"]
                logits = outputs["logits"]                
                
                valid_losses.append(loss.item())
                
                pred_tags = torch.argmax(logits, dim=-1)
                score = calculate_sc_score(true_y=label_tensor.float().tolist(), pred_y=pred_tags.tolist())
                
                
                valid_f1_scores.append(score["f1"])
                valid_acc_scores.append(score["accuracy"])
                
                avg_valid_loss = sum(valid_losses) / len(valid_losses)    
                avg_valid_f1_score = sum(valid_f1_scores) / len(valid_f1_scores)    
                avg_valid_acc_score = sum(valid_acc_scores) / len(valid_acc_scores)    
                
            return avg_valid_loss, avg_valid_f1_score, avg_valid_acc_score
    
    def test(self):
        pass
    
    def predict(self, **parameters):
        
        if "sentence" not in parameters.keys() or "question" not in parameters.keys():
            raise KeyError("The machine reading comprehension task must need sentence and question parameters")
        
        sentence = parameters["sentence"]
        question = parameters["question"]
        
        self.load_model(path=os.path.join(self.model_hub_path, "mrc.mdl"))
        self.model.eval()
        
        with torch.no_grad():
            if type(question) == str:
                question = [question]
                
            token_ids_list, token_type_ids_list = [], []
            for sent, quest in [(sentence, quest) for quest in question]:
                sentence_tokens = self.tokenizer.tokenize(sent)
                question_tokens = self.tokenizer.tokenize(quest)
                
                token_list = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]
                len_question_token_list = len(token_list)
                
                if len(sentence_tokens) > self.max_seq_len - len_question_token_list:
                    sentence_tokens = sentence_tokens[:self.max_seq_len - len_question_token_list]
                else:
                    sentence_tokens = sentence_tokens + [self.tokenizer.pad_token] * (self.max_seq_len - len_question_token_list - len(sentence_tokens))
                
                token_list = token_list + sentence_tokens
                
                token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
                token_type_ids = [0] * len([self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]) + [1] * len(sentence_tokens)
                
                token_ids_list.append(token_ids)
                token_type_ids_list.append(token_type_ids)
                
            token_tensor, token_type_ids_tensor = torch.tensor(token_ids_list), torch.tensor(token_type_ids_list)
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
            
            pred_begin_indexes = torch.argmax(outputs["start_logits"], dim=-1)
            pred_end_indexes = torch.argmax(outputs["end_logits"], dim=-1)
                
            answers = []
            for pred_begin_index, pred_end_index in zip(pred_begin_indexes, pred_end_indexes):
                answers.append(self.tokenizer.convert_tokens_to_string(token_list[pred_begin_index: pred_end_index + 1]))
            
            return answers
        
import os
import torch
from tqdm import tqdm

from .neural_based_task import NeuralBaseTask

from brokorli.metrics.f1_score import calculate_sl_f1_score
from brokorli.special_tokens import LABEL_PAD_TOKEN

class NER(NeuralBaseTask):

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
            
            avg_valid_loss, avg_valid_f1_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss:.4f}\tValid Loss : {avg_valid_loss:.4f}\tValid F1 Score : {avg_valid_f1_score * 100:.4f}")
            
            if max_score < avg_valid_f1_score:
                self.update_trained_model(self.MODEL_PATH.format(epoch, avg_valid_f1_score * 100))
                max_score = avg_valid_f1_score
                
    def valid(self):
        self.model.eval()
        
        with torch.no_grad():
            valid_losses, valid_f1_scores = [], []
            avg_valid_loss, avg_valid_f1_score = 0, 0
            
            progress_bar = tqdm(self.config.valid_data_loader)
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
                
                pred_tags = torch.argmax(logits, dim=-1)
                
                valid_losses.append(loss.item())
                true_y, pred_y = self.decode(label_ids.tolist(), pred_tags.tolist())
                
                score = calculate_sl_f1_score(true_y, pred_y)
                        
                valid_f1_scores.append(score)
                avg_valid_loss = sum(valid_losses) / len(valid_losses)
                avg_valid_f1_score = sum(valid_f1_scores) / len(valid_f1_scores)
            
            return avg_valid_loss, avg_valid_f1_score
    
    def predict(self, **parameters):
        
        if "sentence" not in parameters.keys():
            raise KeyError("The named entity recognition task must need sentence parameter")
        
        sentence = parameters["sentence"]
        
        with torch.no_grad():
            
            if type(sentence) == str:
                sentence = [sentence]
                
            tokenized = self.tokenizer(sentence, return_offsets_mapping=True, padding="max_length", max_length=self.config.max_seq_len, truncation=True)
            
            input_ids, token_type_ids, attention_mask = (
                torch.tensor(tokenized["input_ids"]).to(self.device), 
                torch.tensor(tokenized["token_type_ids"]).to(self.device),
                torch.tensor(tokenized["attention_mask"]).to(self.device)
            )
                            
            outputs = self.model(
                input_ids=input_ids, 
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=None,
            )
            
            entities_list = []
            for idx, output in enumerate(torch.argmax(outputs["logits"], dim=-1).tolist()):
                entities = self.label2entity(
                    token_list=self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][idx]),
                    label_list=[self.i2l[tag] for tag in output], 
                    offset_mapping=tokenized["offset_mapping"][idx]
                )
                
                entities_list.append(entities)
                
            return entities_list
        
    def decode(self, labels, pred_tags):
        true_y_list = []
        pred_y_list = []

        for idx, label in enumerate(labels):
            true_y = []
            pred_y = []

            for jdx in range(len(label)):
                if label[jdx] == self.l2i[LABEL_PAD_TOKEN]:
                    break
                
                true_y.append(self.i2l[label[jdx]])
                pred_y.append("O" if pred_tags[idx][jdx] == self.l2i[LABEL_PAD_TOKEN] else self.i2l[pred_tags[idx][jdx]])
                
            true_y_list.append(true_y)
            pred_y_list.append(pred_y)

        return true_y_list, pred_y_list
    
    def label2entity(self, token_list, label_list, offset_mapping):
        entities = []
        entity_buffer = []
        start_idx, end_idx = 0, 0
        
        for idx, (token, label) in enumerate(zip(token_list, label_list)):
            if label == LABEL_PAD_TOKEN:
                break
            
            if token == self.tokenizer.sep_token:
                break
            
            if label == "O":
                continue
            
            if label.startswith("S-"):
                entities.append({
                    "label": label[2:],
                    "start_idx": offset_mapping[idx][0],
                    "end_idx": offset_mapping[idx][1] - 1,
                })
            
            elif label.startswith("B-") or label.startswith("I-"):                
                if label.startswith("B-"):
                    start_idx = offset_mapping[idx][0]
                    
            elif label.startswith("E-"):
                end_idx = offset_mapping[idx][1]
                
                entities.append({
                    "label": label[2:],
                    "start_idx": start_idx,
                    "end_idx": end_idx - 1
                })
                
                entity_buffer.clear()
                
        return entities
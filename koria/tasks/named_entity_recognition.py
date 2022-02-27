import os
from re import L
import torch
from tqdm import tqdm

from .neural_based_task import NeuralBaseTask

from koria.metrics.f1_score import calculate_sl_f1_score

class NER(NeuralBaseTask):

    def __init__(self, **parameters):
        self.vocab_size = parameters["vocab_size"]
        self.label_size = parameters["label_size"]
        
        self.token_pad_id = parameters["token_pad_id"]
        self.l2i = parameters["l2i"]
        self.i2l = parameters["i2l"]
        
        if "special_label_tokens" in parameters:
            self.special_label_tokens = parameters["special_label_tokens"]
        else:
            self.special_label_tokens = {
                "pad": "[PAD]"
            }
        
        # Set optional parameter for SRL Task
        MODEL_INIT_PARAMETERS = {
            "vocab_size": self.vocab_size,
            "label_size": self.label_size
        }
        
        # Set common parameters for TaskBase and build model
        super().__init__(model_parameters=MODEL_INIT_PARAMETERS, **parameters)
        
    def train(self):
        
        max_score = 0
        
        for epoch in range(int(self.cfg.epochs)):
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
                    attention_mask=(token_tensor != self.token_pad_id).float(),
                    labels=label_tensor,
                )
                
                loss = outputs[0]
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
            
            avg_valid_loss, avg_valid_f1_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss:.4f}\tValid Loss : {avg_valid_loss:.4f}\tValid F1 Score : {avg_valid_f1_score * 100:.4f}")
            
            if max_score < avg_valid_f1_score:
                self.save_model(path=os.path.join(self.model_hub_path, f"ner-e{epoch}-{avg_valid_f1_score * 100:.4f}-lr{self.cfg.learning_rate}-len{self.max_seq_len}.mdl"))
                
    def valid(self):
        self.model.eval()
        
        with torch.no_grad():
            valid_losses, valid_f1_scores = [], []
            avg_valid_loss, avg_valid_f1_score = 0, 0
            
            progress_bar = tqdm(self.valid_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Validation] Avg Loss : {avg_valid_loss:.4f} Avg Score : {avg_valid_f1_score * 100:.4f}")
                
                token_tensor, token_type_ids_tensor, label_tensor = data
                
                if self.use_cuda:
                    token_tensor, token_type_ids_tensor, label_tensor = token_tensor.cuda(), token_type_ids_tensor.cuda(), label_tensor.cuda()
                
                outputs = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=(token_tensor != self.token_pad_id).float(),
                    labels=label_tensor,
                )
                
                loss = outputs[0]
                logits = outputs[1]
                
                pred_tags = torch.argmax(logits, dim=-1)
                
                valid_losses.append(loss.item())
                true_y, pred_y = self.decode(label_tensor, pred_tags)
                
                score = calculate_sl_f1_score(true_y, pred_y)
                        
                valid_f1_scores.append(score)
                
                avg_valid_loss = sum(valid_losses) / len(valid_losses)    
                avg_valid_f1_score = sum(valid_f1_scores) / len(valid_f1_scores)    
                
            return avg_valid_loss, avg_valid_f1_score
    
    def test(self):
        pass
    
    def predict(self, **parameters):
        
        if "sentence" not in parameters.keys():
            raise KeyError("The named entity recognition task must need sentence parameter")
        
        sentence = parameters["sentence"]
        
        self.load_model(path=os.path.join(self.model_hub_path, "ner.mdl"))
        self.model.eval()
        
        with torch.no_grad():
            
            if type(sentence) == str:
                sentence = [sentence]
                
            sent_len_list, token_ids_list, token_type_ids_list = [], [], []
            for sent in sentence:
                sentence_tokens = self.tokenizer.tokenize(sent)
                sent_len_list.append(len(sentence_tokens))
                
                token_list = [self.tokenizer.cls_token] + sentence_tokens + [self.tokenizer.sep_token]
                
                if len(token_list) > self.max_seq_len:
                    token_list = token_list[:self.max_seq_len]
                else:
                    token_list = token_list + [self.tokenizer.pad_token] * (self.max_seq_len - len(sentence_tokens))
                
                token_ids = self.tokenizer.convert_tokens_to_ids(token_list)
                token_type_ids = [1] * len(token_list)
                
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
                labels=None,
            )
            
            entities_list = []
            for idx, output in enumerate(outputs["logits"]):
                label_list = [self.i2l[tag] for tag in torch.argmax(output, dim=-1).tolist()[1:sent_len_list[idx]]]
                entities_list.append(self.label2entity(token_list=token_list[1:], label_list=label_list))
                        
            return entities_list
        
    def decode(self, labels, pred_tags):
        true_y = []
        pred_y = []

        for idx, label in enumerate(labels):
            true = []
            pred = []

            for jdx in range(len(label)):
                if label[jdx] == self.l2i[self.special_label_tokens["pad"]]:
                    break
                
                true.append(self.i2l[label[jdx].item()])
                
                if pred_tags[idx][jdx] == self.l2i[self.special_label_tokens["pad"]]:
                    pred.append("O")
                else:
                    pred.append(self.i2l[pred_tags[idx][jdx].item()])

            true_y.append(true)
            pred_y.append(pred)

        return true_y, pred_y
    
    def label2entity(self, token_list, label_list):
        entities = []
        entity_buffer = []
        for idx, label in enumerate(label_list):
            if label == self.special_label_tokens["pad"]:
                break
            
            if label == "O":
                continue
            
            if label.startswith("S-"):
                entities.append({
                    "entity": token_list[idx],
                    "label": label[2:]
                })
            
            elif label.startswith("B-") or label.startswith("I-"):
                entity_buffer.append(token_list[idx])
            elif label.startswith("E-"):
                entity_buffer.append(token_list[idx])
                entities.append({
                    "entity": self.tokenizer.convert_tokens_to_string(entity_buffer),
                    "label": label[2:]
                })
                
                entity_buffer.clear()
                
        return entities
            
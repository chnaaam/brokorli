import os
import torch
from tqdm import tqdm

from .task_base import TaskBase

from datagene.metrics.f1_score import calculate_f1_score

class SRL(TaskBase):

    def __init__(self, **parameters):
        
        # Set common parameters for TaskBase
        super().__init__(**parameters)
        
        self.vocab_size = parameters["vocab_size"]
        self.token_pad_id = parameters["token_pad_id"]
        self.l2i = parameters["l2i"]
        self.i2l = parameters["i2l"]
        self.special_label_tokens = parameters["special_label_tokens"]
        
        # Set optional parameter for SRL Task
        LAYER_INIT_PARAMETERS = {
            "kobert": {
                "vocab_size": parameters["vocab_size"]    
            },
            "crf": {
                "in_features": self.cfg.crf_in_features,
                "label_size": parameters["label_size"]
            }
        }
        
        self.build(layer_parameters=LAYER_INIT_PARAMETERS)
        
    def train(self):
        
        max_score = 0
        if self.use_cuda:
            self.model = self.model.cuda()
        
        for epoch in range(int(self.cfg.epochs)):
            self.model.train()
            train_losses = []
            avg_train_loss = 0
            
            progress_bar = tqdm(self.train_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Training] Epoch : {epoch}, Avg Loss : {avg_train_loss:.4f}")
                
                token_tensor, token_type_ids_tensor, label_tensor = data
                
                if self.use_cuda:
                    token_tensor, token_type_ids_tensor, label_tensor = token_tensor.cuda(), token_type_ids_tensor.cuda(), label_tensor.cuda()
                
                self.optimizer.zero_grad()
                
                loss, _ = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=(token_tensor != self.token_pad_id).float(),
                    labels=label_tensor,
                    crf_masks=(token_tensor != self.token_pad_id)
                )
                
                if not self.use_deepspeed_lib:
                    loss.backward()
                    self.optimizer.step()
                
                else:
                    self.model.backward(loss)
                    self.model.step()                
                
                train_losses.append(loss.item())
                
                avg_train_loss = sum(train_losses) / len(train_losses)
            
            if not self.use_deepspeed_lib:
                self.scheduler.step()
            
            avg_valid_loss, avg_valid_f1_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss:.4f}\tValid Loss : {avg_valid_loss:.4f}\tValid F1 Score : {avg_valid_f1_score * 100:.2f}")
            
            if max_score < avg_valid_f1_score:
                self.save_model(path=os.path.join(self.model_hub_path, f"srl-score-{avg_valid_f1_score * 100:.2f}-e{epoch}.mdl"))
                
    def valid(self):
        self.model.eval()
        
        with torch.no_grad():
            valid_losses, valid_f1_scores = [], []
            avg_valid_loss, avg_valid_f1_score = 0, 0
            
            progress_bar = tqdm(self.valid_data_loader)
            for data in progress_bar:
                progress_bar.set_description(f"[Validation] Avg Loss : {avg_valid_loss:.4f} Avg Score : {avg_valid_f1_score * 100:.2f}")
                
                token_tensor, token_type_ids_tensor, label_tensor = data
                
                if self.use_cuda:
                    token_tensor, token_type_ids_tensor, label_tensor = token_tensor.cuda(), token_type_ids_tensor.cuda(), label_tensor.cuda()
                
                loss, pred_tags = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=(token_tensor != self.token_pad_id).float(),
                    labels=label_tensor,
                    crf_masks=(token_tensor != self.token_pad_id)
                )
                
                valid_losses.append(loss.item())
                true_y, pred_y = self.decode(label_tensor, pred_tags)
                
                score = calculate_f1_score(true_y, pred_y)
                        
                valid_f1_scores.append(score)
                
                avg_valid_loss = sum(valid_losses) / len(valid_losses)    
                avg_valid_f1_score = sum(valid_f1_scores) / len(valid_f1_scores)    
                
            return avg_valid_loss, avg_valid_f1_score
    
    def test(self):
        pass
    
    def predict(self):
        pass
        
    def decode(self, labels, pred_tags):
        true_y = []
        pred_y = []

        for idx, label in enumerate(labels):
            true = []
            pred = []

            for jdx in range(len(label)):

                if label[jdx] == self.l2i[self.special_label_tokens["begin"]]:
                    continue

                if label[jdx] == self.l2i[self.special_label_tokens["end"]]:
                    break

                if pred_tags[idx][jdx] in [
                    self.l2i[self.special_label_tokens["begin"]], 
                    self.l2i[self.special_label_tokens["pad"]], 
                    self.l2i[self.special_label_tokens["end"]]
                    ]:
                    
                    pred_tags[idx][jdx] = self.l2i["O"]

                true.append(self.i2l[label[jdx].item()])
                pred.append(self.i2l[pred_tags[idx][jdx]])

            true_y.append(true)
            pred_y.append(pred)

        return true_y, pred_y
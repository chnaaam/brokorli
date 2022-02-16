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
        
        super().build(layer_parameters=LAYER_INIT_PARAMETERS)
        
    def train(self):
        
        max_score = 0
        if self.use_cuda:
            self.model = self.model.cuda()
        
        for epoch in range(int(self.cfg.epochs)):
            self.model.train()
            train_loss = []
            
            for data in tqdm(self.train_data_loader, desc=f"Train Epoch : {epoch}"):
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
                
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                
            avg_train_loss = sum(train_loss) / len(train_loss)
            avg_valid_loss, avg_valid_f1_score = self.valid()
            
            print(f"Epoch : {epoch}\tTrain Loss : {avg_train_loss}\tValid Loss : {avg_valid_loss}\tValid F1 Score : {avg_valid_f1_score}")
            
            if max_score < avg_valid_f1_score:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_hub_path, f"srl-score-{avg_valid_f1_score:.2f}-e{epoch}.mdl")
                )
                
    def valid(self):
        valid_loss, valid_f1_score = [], []
        
        self.model.eval()
        
        with torch.no_grad():
            for data in tqdm(self.valid_data_loader, desc=f"Validation : "):
                token_tensor, token_type_ids_tensor, label_tensor = data
                
                if self.use_cuda:
                    token_tensor, token_type_ids_tensor, label_tensor = token_tensor.cuda(), token_type_ids_tensor.cuda(), label_tensor.cuda()
                
                self.optimizer.zero_grad()
                
                loss, pred_tags = self.model(
                    token_tensor, 
                    token_type_ids=token_type_ids_tensor,
                    attention_mask=(token_tensor != self.token_pad_id).float(),
                    labels=label_tensor,
                    crf_masks=(token_tensor != self.token_pad_id)
                )
                
                valid_loss.append(loss.item())
                true_y, pred_y = self.decode(label_tensor, pred_tags)
                
                score = calculate_f1_score(true_y, pred_y)
                        
                valid_f1_score.append(score)        
                
            return sum(valid_loss) / len(valid_loss), sum(valid_f1_score) / len(valid_f1_score)
    
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
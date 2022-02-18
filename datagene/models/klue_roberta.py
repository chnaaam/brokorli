
import torch.nn as nn

from transformers import RobertaModel

class KlueRobertaLayer(nn.Module):
    def __init__(self, **parameters):
        super().__init__()
        
        self.roberta = RobertaModel.from_pretrained("klue/roberta-base")
        self.roberta.resize_token_embeddings(parameters["vocab_size"])
        
    def forward(self, X, parameters):
        token_type_ids = parameters["token_type_ids"]
        attention_mask = parameters["attention_mask"]
        
        features = self.roberta(
            X,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        return features[0]
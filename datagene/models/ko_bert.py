import torch.nn as nn

from transformers import BertModel

class KoBERTLayer(nn.Module):
    def __init__(self, **parameters):
        super().__init__()
        
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.bert.resize_token_embeddings(parameters["vocab_size"])
        
    def forward(self, X, parameters):
        token_type_ids = parameters["token_type_ids"]
        attention_mask = parameters["attention_mask"]
        
        features = self.bert(
            X,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        return features[0]
import torch.nn as nn

from transformers import ElectraModel

class KoElectraLayer(nn.Module):
    def __init__(self, **parameters):
        super().__init__()
        
        self.electra = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.electra.resize_token_embeddings(parameters["vocab_size"])
        
    def forward(self, X, parameters):
        token_type_ids = parameters["token_type_ids"]
        attention_mask = parameters["attention_mask"]
        
        features = self.electra(
            X,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        return features[0]
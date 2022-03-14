import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForSequenceClassification

class SequenceClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels=None, vocab_size=None):
        super().__init__()
        
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        if vocab_size:
            self.model.resize_token_embeddings(vocab_size)
        
    def forward(self, **parameters):
        input_ids = parameters["input_ids"]
        token_type_ids = parameters["token_type_ids"]
        attention_mask = parameters["attention_mask"]
        labels = parameters["labels"]
        
        # TODO: Roberta does not need token_type_ids
        if "roberta" not in self.model_name:
            outputs = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
        return outputs
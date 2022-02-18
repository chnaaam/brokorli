import torch.nn as nn

from transformers import AutoConfig, AutoModelForTokenClassification

class SequenceLabelingModel(nn.Module):
    def __init__(self, model_name, parameters):
        super().__init__()
        
        self.model_name = model_name
        
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = parameters["label_size"]
        
        self.model = AutoModelForTokenClassification.from_config(config)
        self.model.resize_token_embeddings(parameters["vocab_size"])
        
    def forward(self, X, **parameters):
        token_type_ids = parameters["token_type_ids"]
        attention_mask = parameters["attention_mask"]
        labels = parameters["labels"]
        
        # TODO: Roberta does not need token_type_ids
        if "roberta" not in self.model_name:
            outputs = self.model(
                X,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = self.model(
                X,
                attention_mask=attention_mask,
                labels=labels,
            )
            
        return outputs
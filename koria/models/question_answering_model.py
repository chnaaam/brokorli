import torch.nn as nn

from transformers import AutoConfig, AutoModelForQuestionAnswering

class QuestionAnsweringModel(nn.Module):
    def __init__(self, model_name, num_labels=None, vocab_size=None):
        super().__init__()
        
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        
        if num_labels:
            config.num_labels = num_labels    
        
        self.model = AutoModelForQuestionAnswering.from_config(config)
        
        if vocab_size:
            self.model.resize_token_embeddings(vocab_size)
        
    def forward(self, **parameters):
        input_ids = parameters["input_ids"]
        token_type_ids = parameters["token_type_ids"]
        attention_mask = parameters["attention_mask"]
        start_positions = parameters["start_positions"]
        end_positions = parameters["end_positions"]
        
        # TODO: Roberta does not need token_type_ids
        if "roberta" not in self.model_name:
            outputs = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            
        return outputs
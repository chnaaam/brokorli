import torch.nn as nn

from torchcrf import CRF

class CRFLayer(nn.Module):
    def __init__(self, **parameters):
        super().__init__()
        
        self.fc = nn.Linear(in_features=parameters["in_features"], out_features=parameters["label_size"])
        self.crf = CRF(num_tags=parameters["label_size"], batch_first=True)
        
    def forward(self, X, parameters):
        labels = parameters["labels"]
        crf_masks = parameters["crf_masks"]
        
        emissions = self.fc(X)

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, crf_masks)
            sequence_of_tags = self.crf.decode(emissions, crf_masks)

            return (-1) * log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, crf_masks)

            return sequence_of_tags
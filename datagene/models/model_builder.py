import torch
import torch.nn as nn

class ModelBuilder(nn.Module):
    def __init__(self, layer_list, layer_parameters):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for layer_name, layer in layer_list.items():
            self.layers.append(layer(**layer_parameters[layer_name]))
        
    def forward(self, X, **parameters):
        for layer in self.layers:
            X = layer(X, parameters)
        
        return X
        
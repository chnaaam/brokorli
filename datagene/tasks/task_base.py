from abc import *

from datagene.models import ModelBuilder
import torch.nn as nn

class TaskBase(metaclass=ABCMeta):
    def __init__(self, **parameters):
        self.cfg = parameters["parameter_cfg"]
        
        self.layer_list = parameters["layer_list"]
        self.tokenizer = parameters["tokenizer"]
        
        self.train_data_loader = parameters["data_loader"]["train"]
        self.valid_data_loader = parameters["data_loader"]["valid"]
        self.test_data_loader = parameters["data_loader"]["test"]
        
        self.optimizer_func = parameters["optimizer"]
        self.criterion = parameters["criterion"]()
        
        self.use_cuda = False
        if "use_cuda" in parameters:
            self.use_cuda = parameters["use_cuda"]
           
            
    def build(self, layer_parameters):
        self.model = ModelBuilder(layer_list=self.layer_list, layer_parameters=layer_parameters)
        self.optimizer = self.optimizer_func(self.model.parameters(), lr=float(self.cfg.learning_rate))
        
        if self.use_cuda:
            self.model = self.model.cuda()
            
    @abstractmethod
    def train(self):
        pass
            
    @abstractmethod
    def predict(self):
        pass
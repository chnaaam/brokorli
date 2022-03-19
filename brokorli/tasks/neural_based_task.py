from abc import *

import os
import pickle
import torch
from accelerate import Accelerator

from brokorli.models import MODEL_LIST


class NeuralBaseTask(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config
        
        self.l2i, self.i2l = None, None
        if config.label_hub_path and os.path.isfile(os.path.join(config.label_hub_path, f"{config.task_name}.label")):
            with open(os.path.join(config.label_hub_path, f"{config.task_name}.label"), "rb") as fp:
                self.l2i = pickle.load(fp)["l2i"]
                self.i2l = {i: l for l, i in self.l2i.items()}
        
        self.tokenizer = config.tokenizer
        self.accelerator = Accelerator(fp16=self.config.use_fp16, cpu=False if self.config.use_cuda else True)
        self.device = self.accelerator.device
            
        if not config.train_data_loader:
            self.model = MODEL_LIST[config.task_name](
                model_name=config.pretrained_model_name,
                num_labels=len(self.l2i) if self.l2i else None
            )
            
            self.model = self.model.to(self.device)
        else:
            self.model = MODEL_LIST[self.config.task_name](
                model_name=self.config.pretrained_model_name,
                num_labels=len(self.l2i) if self.l2i else None
            )
            
            self.optimizer = self.config.optimizer(self.model.parameters(), lr=float(self.config.learning_rate))
            self.model, self.optimizer, self.train_data_loader = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.config.train_data_loader
            )
            
            self.MODEL_PATH = self.config.task_name + ".e{0}.score{1:02.2f}" + f".lr{self.config.learning_rate}.len{self.config.max_seq_len}.mdl"
            self.previous_model_path = ""
        
    def update_trained_model(self, model_name):
        if self.previous_model_path:
            if os.path.isfile(os.path.join(self.config.model_hub_path, self.previous_model_path)):
                os.remove(os.path.join(self.config.model_hub_path, self.previous_model_path))
                
        self.save_model(os.path.join(self.config.model_hub_path, model_name))
        self.previous_model_path = model_name
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def valid(self):
        pass
            
    @abstractmethod
    def predict(self, **parameters):
        pass
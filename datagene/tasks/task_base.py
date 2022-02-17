from abc import *

import os
import torch

from datagene.models import ModelBuilder

class TaskBase(metaclass=ABCMeta):
    """
    Task에 대한 추상 클래스 입니다. 
    만약, 새로운 Task를 정의하는 경우, 해당 클래스를 상속 받아 사용하면 됩니다.
    """
    def __init__(self, **parameters):
        """
        기본 Task 설정에 필요한 코드이며, 아래 변수들은 고정으로 사용됩니다.
        """
        
        # Model parameter
        self.cfg = parameters["parameter_cfg"]
        
        # Layer list for building model
        self.layer_list = parameters["layer_list"]
        self.tokenizer = parameters["tokenizer"]
        
        # Data loader
        self.train_data_loader = parameters["data_loader"]["train"]
        self.valid_data_loader = parameters["data_loader"]["valid"]
        self.test_data_loader = parameters["data_loader"]["test"]
        
        # Optimizer
        self.optimizer_func = parameters["optimizer"]
        
        # Loss function
        self.criterion = parameters["criterion"]()
        
        # Scheduler
        self.scheduler_func = parameters["scheduler"]
        
        # Use gpu
        self.use_cuda = False
        if "use_cuda" in parameters:
            self.use_cuda = parameters["use_cuda"]
            
        self.model_hub_path = parameters["model_hub_path"]
           
            
    def build(self, layer_parameters):
        """
        모델 및 Optimizer를 설정하는 함수로, Configuration 파일 내 모델 이름에 따라 layer_paremeter의 값을 설정해주면 됩니다.
        
        layer_parameters argument example when model name is "kobert-crf"
        
        layer_parameters = {
            "kobert": {
                "vocab_size": parameters["vocab_size"]
            },
            "crf": {
                "in_features": self.cfg.crf_in_features,
                "label_size": parameters["label_size"]
            }
        }
        
        super().build(layer_parameters=layer_parameters)
        """
        self.model = ModelBuilder(layer_list=self.layer_list, layer_parameters=layer_parameters)
        
        param_optimizer = list(self.model.parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.cfg.parameters.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        self.optimizer = self.optimizer_func(
            # self.model.parameters(), 
            optimizer_grouped_parameters,
            lr=float(self.cfg.learning_rate)
        )
        
        # TODO : Please edit the parameter. I wrote lambda scheduler hard code for the current version
        lambda1 = lambda epoch: self.cfg.epochs // 2
        lambda2 = lambda epoch: 0.95 ** self.cfg.epochs
        
        self.scheduler = self.scheduler_func(
            optimizer=self.optimizer,
            lr_lambda = [lambda1, lambda2]
        )
        
        # If "use_cuda" parameter of configuration file is True, model is trained using gpu
        if self.use_cuda:
            self.model = self.model.cuda()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    @abstractmethod
    def train(self):
        """
        train 함수는 모델 학습을 위해 사용됩니다.
        TaskBase 클래스를 상속한 경우, 해당 추상 함수를 반드시 정의해주세요.
        """
        pass
           
    @abstractmethod
    def valid(self):
        """
        valid 함수는 모델 학습을 위해 사용됩니다.
        TaskBase 클래스를 상속한 경우, 해당 추상 함수를 반드시 정의해주세요.
        """
        pass
            
    @abstractmethod
    def test(self):
        """
        test 함수는 모델 학습을 위해 사용됩니다.
        TaskBase 클래스를 상속한 경우, 해당 추상 함수를 반드시 정의해주세요.
        """
        pass
            
    @abstractmethod
    def predict(self):
        """
        train 함수는 모델 학습을 위해 사용됩니다.
        TaskBase 클래스를 상속한 경우, 해당 추상 함수를 반드시 정의해주세요.
        """
        pass
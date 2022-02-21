from abc import *

import torch
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

from koria.models import MODEL_LIST


class TaskBase(metaclass=ABCMeta):
    """
    Task에 대한 추상 클래스 입니다. 
    만약, 새로운 Task를 정의하는 경우, 해당 클래스를 상속 받아 사용하면 됩니다.
    """
    def __init__(self, model_parameters, **parameters):
        """
        기본 Task 설정에 필요한 코드이며, 아래 변수들은 고정으로 사용됩니다.
        """
        
        # Model parameter
        self.cfg = parameters["parameter_cfg"]
        
        # Data loader
        self.train_data_loader = parameters["data_loader"]["train"]
        self.valid_data_loader = parameters["data_loader"]["valid"]
        self.test_data_loader = parameters["data_loader"]["test"]
        
        # Optimizer
        self.optimizer_func = parameters["optimizer"]
      
        # Use gpu
        self.use_cuda = False
        if "use_cuda" in parameters:
            self.use_cuda = parameters["use_cuda"]
                    
        # Hub Path for Trained Model
        self.model_hub_path = parameters["model_hub_path"]
        
        # Huggingface Accelerator
        self.accelerator = Accelerator(
            fp16=self.cfg.fp16, 
            cpu=False if self.use_cuda else True
        )
        
        # LM Model
        self.model = MODEL_LIST[parameters["model_type"]](
            model_name=parameters["model_name"], 
            parameters=model_parameters
        )
        
        self.tokenizer = parameters["tokenizer"]
        
        self.optimizer = parameters["optimizer"](
            self.model.parameters(),
            lr=float(self.cfg.learning_rate)
        )
                
        self.device = self.accelerator.device
        
        self.model.to(self.device)
        
        self.model, self.optimizer, self.train_data_loader = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_data_loader
        )
            
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
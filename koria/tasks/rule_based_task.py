import os
import yaml

from abc import *

class RuleBasedTask(metaclass=ABCMeta):
    """
    규칙 기반의 Task에 대한 추상 클래스 입니다. 
    만약, 새로운 규칙 기반 Task를 정의하는 경우, 해당 클래스를 상속 받아 사용하면 됩니다.
    """
    def __init__(self, **parameters):
        self.cfg = parameters["cfg"]
        self.template_dir = parameters["template_dir"]
                
        if self.cfg.template.path not in os.listdir(self.template_dir):
            raise FileNotFoundError(f"Template directory is not existed in {os.path.abspath(os.path.join(self.template_dir, self.cfg.template.path))}")
        
        template_file_list = os.listdir(os.path.join(self.template_dir, self.cfg.template.path))
        
        if not template_file_list:
            raise ValueError(f"Template files are not existed in {os.path.abspath(os.path.join(self.template_dir, self.cfg.template.path))}")
        
        self.templates = dict()
        for template_fn in template_file_list:
            with open(os.path.join(self.template_dir, self.cfg.template.path, template_fn), "r", encoding="utf-8") as fp:
                template = yaml.load(fp, Loader=yaml.FullLoader)
                
            if template:
                self.templates.setdefault(template_fn.split(".")[0], template)
    
    @abstractmethod
    def predict(self, **parameters):
        """
        train 함수는 모델 학습을 위해 사용됩니다.
        TaskBase 클래스를 상속한 경우, 해당 추상 함수를 반드시 정의해주세요.
        """
        pass
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
        self.rule_dir = parameters["rule_dir"]
                
        if self.cfg.template.path not in os.listdir(self.rule_dir):
            raise FileNotFoundError(f"Rule directory is not existed in {os.path.abspath(os.path.join(self.rule_dir, self.cfg.template.path))}")
        
        rule_list = os.listdir(os.path.join(self.rule_dir, self.cfg.template.path))
        
        if not rule_list:
            raise ValueError(f"Rule files are not existed in {os.path.abspath(os.path.join(self.rule_dir, self.cfg.template.path))}")
        
        self.rules = dict()
        for rule_fn in rule_list:
            with open(os.path.join(self.rule_dir, self.cfg.template.path, rule_fn), "r", encoding="utf-8") as fp:
                rule = yaml.load(fp, Loader=yaml.FullLoader)
                
            if rule:
                self.rules.setdefault(rule_fn.split(".")[0], rule)
    
    def convert_josa():
        # 1. 맨 뒷 글자가 영어인지 아닌지 판별
        
        # 2. 이 / 가, 은 / 는 구별
        pass
    
    @abstractmethod
    def predict(self, **parameters):
        """
        train 함수는 모델 학습을 위해 사용됩니다.
        TaskBase 클래스를 상속한 경우, 해당 추상 함수를 반드시 정의해주세요.
        """
        pass
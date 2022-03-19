import os
import yaml

from abc import *


class UserTemplate:
    def __init__(self, subj_type, relation, obj_types, templates):
        self.subj_type = subj_type
        self.relation = relation
        self.obj_types = obj_types
        self.templates = templates

class RuleBasedTask(metaclass=ABCMeta):
    """
    규칙 기반의 Task에 대한 추상 클래스 입니다. 
    만약, 새로운 규칙 기반 Task를 정의하는 경우, 해당 클래스를 상속 받아 사용하면 됩니다.
    """
    def __init__(self, config):
        self.config = config
        self.template_dir = config.template_dir
                
        template_file_list = os.listdir(self.template_dir)
        
        if not template_file_list:
            raise ValueError(f"Template files are not existed in {os.path.abspath(os.path.join(self.template_dir, self.cfg.template.path))}")
        
        self.templates = dict()
        for template_fn in template_file_list:
            with open(os.path.join(self.template_dir, template_fn), "r", encoding="utf-8") as fp:
                template = yaml.load(fp, Loader=yaml.FullLoader)
                
            if template:
                self.templates.setdefault(template_fn.split(".")[0], template)
    
    def add_templates(self, user_template):
        subj_type, relation, obj_types, templates = user_template.subj_type, user_template.relation, user_template.obj_types, user_template.templates
        
        obj_types = [obj_types] if type(obj_types) == str else obj_types
        templates = [templates] if type(templates) == str else templates
        
        if subj_type not in self.templates:
            self.templates.setdefault(subj_type, {
                relation: {
                    "object_types": obj_types,
                    "templates": templates
                }
            })
        else:
            self.templates[subj_type].setdefault(relation, {
                "obj_types": obj_types,
                "templates": templates
            })
    
    @abstractmethod
    def predict(self, **parameters):
        """
        train 함수는 모델 학습을 위해 사용됩니다.
        TaskBase 클래스를 상속한 경우, 해당 추상 함수를 반드시 정의해주세요.
        """
        pass
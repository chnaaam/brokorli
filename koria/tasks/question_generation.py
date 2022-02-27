from pyjosa.josa import Josa

from .rule_based_task import RuleBasedTask


class QG(RuleBasedTask):

    def __init__(self, **parameters):
        super().__init__(**parameters)
        
        self.ENTITY_TOKEN = "{E}"
   
    def is_registered_entity_type(self, type):
        return type.lower() in self.rules
   
    def predict(self, **parameters):
        
        if "entity" not in parameters.keys() or "type" not in parameters.keys():
            raise KeyError("The question generation task must need entity name and type parameters")
        
        entity = parameters["entity"]
        type = parameters["type"].lower()
        
        if type not in self.rules.keys():
            raise KeyError(f"{type} is not defined in rule files")
        
        rules = self.rules[type]
        
        questions = dict()
        
        for relation, rule in rules.items():
            questions.setdefault(relation, {"obj_types": rule["obj_types"], "questions": []})
            
            for template in rule["templates"]:
                questions[relation]["questions"].append(self.reconstruct_question(template, entity))
        
        return questions
    
    def reconstruct_question(self, template, entity,):
        
        candidate_josa = template.split("_")[1]
        
        # 1. 맨 뒷 글자가 한글인지 아닌지 판별
        if self.is_hangul(entity[-1]):
            # 2. 이 / 가, 은 / 는 구별
            josa = Josa.get_josa(entity, candidate_josa.split("/")[0])
        else:
            # 2. 영어인 경우 앞에것을 임의로 선택
            josa = candidate_josa[0]
            
        template = template.replace(f"_{candidate_josa}_", josa)
        template = template.replace(self.ENTITY_TOKEN, entity)
        
        return template
           
    def is_hangul(self, c):
        if ord('가') <= ord(c) <= ord('힣'):
            return True
        
        return False
        
from pyjosa.josa import Josa

from .rule_based_task import RuleBasedTask


class QG(RuleBasedTask):

    def __init__(self, task_config):
        super().__init__(config=task_config)
        
        self.ENTITY_TOKEN = "{E}"
   
    def is_registered_entity_type(self, type):
        return type.lower() in self.templates
   
    def get_rel_from_defined_templates(self, type):
        return list(self.templates[type].keys())
   
    def predict(self, entity, entity_type, with_entity_marker=""):
        questions = dict()
        entity_type = entity_type.lower()
        
        if entity_type not in self.templates.keys():
            return questions
        
        templates = self.templates[entity_type]
        
        for relation, template in templates.items():
            questions.setdefault(relation, {"obj_types": template["obj_types"], "questions": []})
            
            for template in template["templates"]:
                questions[relation]["questions"].append(self.reconstruct_question(template, entity, with_entity_marker))
        
        return questions
    
    def reconstruct_question(self, template, entity, with_entity_marker):
        
        if "_" in template:
            candidate_josa = template.split("_")[1]
            
            # 1. 맨 뒷 글자가 한글인지 아닌지 판별
            if self.is_hangul(entity[-1]):
                # 2. 이 / 가, 은 / 는 구별
                josa = Josa.get_josa(entity, candidate_josa.split("/")[0])
            else:
                # 2. 영어인 경우 앞에것을 임의로 선택
                josa = candidate_josa[0]
                
            template = template.replace(f"_{candidate_josa}_", josa)
            
        return template.replace(self.ENTITY_TOKEN, f"{with_entity_marker}{entity}{with_entity_marker}")
           
    def is_hangul(self, c):
        if ord('가') <= ord(c) <= ord('힣'):
            return True
        
        return False
    
    # Use for data generation
    # def generate_true_question_using_rel_type(self, entity, type, rel):
    #     type = type.lower()
    #     template = self.templates[type][rel]
        
    #     questions = []
    #     for template in template["templates"]:
    #         questions.append(self.reconstruct_question(template, entity))
        
    #     return questions
    
    # def generate_false_question_using_rel_type(self, entity, type, rel):
    #     type = type.lower()
    #     unselected_rel_list = list(self.templates[type].keys())
    #     unselected_rel_list.remove(rel)
        
    #     import random
    #     rel = random.choice(unselected_rel_list)
        
    #     template = self.templates[type][rel]
        
    #     questions = []
    #     for template in template["templates"]:
    #         questions.append(self.reconstruct_question(template, entity))
        
    #     return questions
from .rule_based_task import RuleBasedTask


class QG(RuleBasedTask):

    def __init__(self, **parameters):
        super().__init__(**parameters)
        
        self.ENTITY_TOKEN = "{E}"
   
    def predict(self, **parameters):
        
        if "entity" not in parameters.keys() or "type" not in parameters.keys():
            raise KeyError("The question generation task must need entity name and type parameters")
        
        entity = parameters["entity"]
        type = parameters["type"].lower()
        
        if type not in self.rules.keys():
            raise KeyError(f"{type} is not defined in rule files")
        
        templates = self.rules[type]
        questions = dict()
        
        for relation, template_list in templates.items():
            questions.setdefault(relation, [])
            
            for template in template_list:
                questions[relation].append(template.replace(self.ENTITY_TOKEN, entity))
                
        print(questions)
        return questions
        
        
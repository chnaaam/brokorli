from . import PathBase


class EntityRecognitionPath(PathBase):
    
    """
    Common Workflow
    Common Workflow는 Named Entity Recognition Task를 이용하여 Entity를 추출하는 작업을 수행합니다.
    해당 플로우를 수행하기 위해서는 아래와 같은 Task 들이 필요합니다.
    - Named Entity Recognition
    """
    
    def __init__(self, tasks):
        self.ner_task = tasks["ner"]
        
    def run(self, sentence):
        return self.ner_task.predict(sentence=sentence)
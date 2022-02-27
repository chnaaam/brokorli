from . import (
    EntityRecognitionPath,
    ZeroShotPath
)

class Workflow:
    def __init__(self, tasks):
        self.tasks = tasks
        
        self.entity_recognition_path = EntityRecognitionPath(tasks)
        self.zero_shot_path = ZeroShotPath(tasks)
        
    def cli(self):
        while True:
            sentence = input(">>> ")
            entities = self.entity_recognition_path.run(sentence=sentence)
            
            triples = self.zero_shot_path.run(sentence=sentence, entities=entities)
        
        
    
    
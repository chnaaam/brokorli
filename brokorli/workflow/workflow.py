from . import (
    EntityRecognitionPath,
    ZeroShotPath
)

class Workflow:
    def __init__(self, tasks):
        self.tasks = tasks
        
        self.entity_recognition_path = EntityRecognitionPath(tasks)
        self.zero_shot_path = ZeroShotPath(tasks)
    
    def run(self, sentence, mrc_threshold=0.9):
        entities = self.entity_recognition_path.run(sentence=sentence)
        triples = self.zero_shot_path.run(sentence=sentence, entities=entities[0], mrc_threshold=mrc_threshold)
        
        return triples
    
    def cli(self):
        while True:
            print(self.run(sentence=input(">>> ")))    
    
    
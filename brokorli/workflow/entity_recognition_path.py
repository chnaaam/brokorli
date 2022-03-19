from . import PathBase


class EntityRecognitionPath(PathBase):
    def __init__(self, tasks):
        self.ner_task = tasks["ner"]
        
    def run(self, sentence):
        return self.ner_task.predict(sentence=sentence)
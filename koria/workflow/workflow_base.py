from abc import *

class WorkflowBase(metaclass=ABCMeta):
    def __init__(self):
        pass
    
    def recognize_entities(self):
        pass
    
    @abstractmethod
    def inference(self, **parameters):
        pass
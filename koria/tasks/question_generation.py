from .task_base import TaskBase


class QG(TaskBase):

    def __init__(self, **parameters):
                
        # Set common parameters for TaskBase and build model
        super().__init__(model_parameters=None, **parameters)
        
    def train(self):
        raise NotImplementedError("Question generation task does not use train function in the current version.")
    
    def valid(self):
        raise NotImplementedError("Question generation task does not use valid function in the current version.")
    
    def test(self):
        raise NotImplementedError("Question generation task does not use test function in the current version.")
    
    def predict(self):
        pass
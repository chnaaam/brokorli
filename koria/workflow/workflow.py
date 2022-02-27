from . import (
    CommonPath,
    ZeroShotPath
)

class Workflow:
    def __init__(self, tasks):
        self.tasks = tasks
        
        self.common_path = CommonPath(tasks)
        self.zero_shot_path = ZeroShotPath(tasks)
        
    def run(self, **parameters):
        pass
    
    
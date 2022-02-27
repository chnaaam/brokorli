from abc import *

class PathBase(metaclass=ABCMeta):
    
    @abstractmethod
    def run(self, **parameters):
        pass
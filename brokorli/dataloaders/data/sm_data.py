from tqdm import tqdm

# from . import DataBase
from brokorli.dataloaders.data.data_base import DataBase

class SmData(DataBase):
    
    """
    Semantic Matching Classification Dataset
    - KorQuad 1.0
    """
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        self.data = self.dump
from tqdm import tqdm

from . import DataBase

class NerData(DataBase):
    
    """
    Named Entity Recognition Dataset
    - 모두의 말뭉치 개체명 분석 말뭉치
    """
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        self.data = self.dump
    
    
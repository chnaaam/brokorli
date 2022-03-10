from tqdm import tqdm

from . import DataBase

class NerData(DataBase):
    
    """
    Named Entity Recognition Dataset
    - 모두의 말뭉치 개체명 분석 말뭉치
    """
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        for document in tqdm(self.dump["document"], desc=f"Load dataset"):
            sentences = document["sentence"]
            
            for sentence in sentences:
                form = sentence["form"]
                ne_list = sentence["NE"]
                
                if form and ne_list:
                    self.data.append({
                        "sentence": form,
                        "entities": ne_list,
                    })
    
    
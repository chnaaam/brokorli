from tqdm import tqdm

# from . import DataBase
from koria.data_loaders.data_list.data_base import DataBase

class MrcData(DataBase):
    
    """
    Machine Reading Comprehension Dataset
    - KorQuad 1.0
    """
    
    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        
        for data in tqdm(self.dump["data"], desc=f"Load dataset"):
            
            for paragraph in data["paragraphs"]:
                context = paragraph["context"]        
                
                for qas in paragraph["qas"]:
                    question = qas["question"]
                    answer = qas["answers"][0]
                    
                    answer_begin = int(answer["answer_start"])
                    answer_end = answer_begin + len(answer["text"]) - 1
                    
                    self.data.append({
                        "context": context,
                        "question": question,
                        "answer": {
                            "begin": answer_begin,
                            "end": answer_end
                        }
                    })
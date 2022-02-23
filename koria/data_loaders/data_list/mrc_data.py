from tqdm import tqdm
from kss import split_sentences

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
                
                sentence_list = split_sentences(context)
                
                for qas in paragraph["qas"]:
                    question = qas["question"]
                    answer = qas["answers"][0]
                    
                    answer_begin = int(answer["answer_start"])
                    answer_end = answer_begin + len(answer["text"]) - 1
                    
                    sent_idx = 0
                    for idx, sentence in enumerate(sentence_list):
                        
                        if sent_idx <= answer_begin <= sent_idx + len(sentence):
                            adjusted_answer_begin = answer_begin - sent_idx
                            adjusted_answer_end = answer_end - sent_idx
                            
                            target_sentence = sentence_list[idx]
                            break
                        
                        sent_idx += len(sentence) + 1
                    
                    assert context[answer_begin: answer_end + 1] == target_sentence[adjusted_answer_begin: adjusted_answer_end], "Adjusted begin and end index is not matched"
                    
                    self.data.append({
                        "sentence": target_sentence,
                        "question": question,
                        "answer": {
                            "begin": adjusted_answer_begin,
                            "end": adjusted_answer_end
                        }
                    })

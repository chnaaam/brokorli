from collections import Counter

from . import PathBase

class ZeroShotPath(PathBase):
    
    """
    MRC Workflow
    MRC Workflow는 Question Generation Task와 Machine Reading Comprehension Task를 이용해서 관계형 정보를 추출합니다.
    해당 플로우를 수행하기 위해서는 아래와 같은 Task 들이 필요합니다.
    - Rule-based Question Generation
    - Question-Sentence Matching Classification (예정)
    - Machine Reading Comprehension
    """
    
    def __init__(self, tasks):
        self.qg_task = tasks["qg"]
        self.mrc_task = tasks["mrc"]
        
    def run(self, **parameters):
        if "sentence" not in parameters.keys() or "entities" not in parameters.keys():
            raise KeyError("The zero-shot path must need sentence and entities parameter")
        
        triple_list = []
        
        sentence = parameters["sentence"]
        entities = parameters["entities"]
        
        for subj_idx, subj_info in enumerate(entities):
            subj = subj_info["entity"]
            subj_type = subj_info["label"]
            
            if not self.qg_task.is_registered_entity_type(subj_type):
                continue
            
            questions = self.qg_task.predict(entity=subj, type=subj_type)
            
            for relation, question_info in questions.items():
                counted_answer = Counter(self.mrc_task.predict(sentence=sentence, question=question_info["questions"]))
                answer = counted_answer.most_common(1)[0][0]
                
                for obj_idx, obj_info in enumerate(entities):
                    if subj_idx == obj_idx:
                        continue
                    
                    obj = obj_info["entity"]
                    obj_type = obj_info["label"].lower()
            
                    if (answer in obj or obj in answer) and obj_type in question_info["obj_types"]:
                        if obj and answer:
                            triple_list.append((subj, relation, min(answer, obj, key=len)))
                        elif obj:
                            triple_list.append((subj, relation, obj))
                        else:
                            triple_list.append((subj, relation, answer))
        
        return triple_list
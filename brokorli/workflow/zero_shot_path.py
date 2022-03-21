from collections import Counter

from . import PathBase

class ZeroShotPath(PathBase):
    def __init__(self, tasks):
        self.qg_task = tasks["qg"]
        self.sm_task = tasks["sm"]
        self.mrc_task = tasks["mrc"]
        
    def run(self, sentence, entities, mrc_threshold):
        triple_list = []
        
        for subj_idx, subj_info in enumerate(entities):
            subj = sentence[subj_info["start_idx"]: subj_info["end_idx"] + 1]
            subj_type = subj_info["label"]
            
            if not self.qg_task.task.is_registered_entity_type(subj_type):
                continue
            
            questions = self.qg_task.predict(entity=subj, entity_type=subj_type)
            
            for relation, question_list in questions.items():
                results = self.sm_task.predict(sentence=sentence, question=question_list["questions"])
                
                true_questions = []
                for idx, result in enumerate(results):
                    if result:
                        true_questions.append(question_list["questions"][idx])
                
                if not true_questions:
                    continue
                
                pred_answers = self.mrc_task.predict(sentence=sentence, question=true_questions)
                answers = []
                
                for pred_answer in pred_answers:
                    answer = pred_answer["answer"]
                    confidence_score = pred_answer["confidence_score"]
                    
                    if confidence_score < mrc_threshold:
                        continue
                    
                    if not answer or answer.startswith("##"):
                        continue
                    
                    for obj_idx, obj_info in enumerate(entities):
                        if subj_idx == obj_idx:
                            continue
                        
                        obj = sentence[obj_info["start_idx"]: obj_info["end_idx"] + 1]
                        obj_type = obj_info["label"].lower()

                        if not obj:
                            continue
                        
                        if (answer in obj or obj in answer) and obj_type in question_list["obj_types"]:
                            if answer == obj:
                                answers.append(answer)
                            else:
                                answers.append(min(answer, obj, key=len))
                
                if answers:
                    answer = Counter(answers).most_common(1)[0][0]
                    
                    triple_list.append((subj, relation, answer))
        
        return triple_list
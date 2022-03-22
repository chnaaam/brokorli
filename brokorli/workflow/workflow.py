class Workflow:
    def __init__(self, tasks):
        self.ner_task = tasks["ner"]
        self.qg_task = tasks["qg"]
        self.sm_task = tasks["sm"]
        self.mrc_task = tasks["mrc"]
        
    def run(self, sentences, sm_batch_size=4, mrc_batch_size=4, mrc_threshold=0.9):
        sentences = sentences if type(sentences) == list else [sentences]
                
        batched_entities = self.recognize_ne(sentences=sentences)
        
        sentence_question_pairs = self.generate_question(sentences=sentences, entities=batched_entities)
        sentence_question_pairs = self.classify_semantic_matching(sentence_question_pairs=sentence_question_pairs, batch_size=sm_batch_size)
        answers = self.get_answer_from_sentence_question_pairs(sentence_question_pairs=sentence_question_pairs, batch_size=mrc_batch_size, mrc_threshold=mrc_threshold)
        
        triples = self.get_triples(entities=batched_entities, answer=answers)
        
        return triples
    
    def recognize_ne(self, sentences):
        return self.ner_task.predict(sentence=sentences)
    
    def generate_question(self, sentences, entities):
        sentence_question_pairs = []
        
        for idx, (sentence, entity_list) in enumerate(zip(sentences, entities)):
            for entity in entity_list:
                entity_start_idx, entity_end_idx = entity["start_idx"], entity["end_idx"]
                named_entity = sentence[entity_start_idx: entity_end_idx + 1]
                entity_type = entity["label"]
                
                if self.qg_task.task.is_registered_entity_type(entity_type):
                    for relation, question_list in self.qg_task.predict(entity=named_entity, entity_type=entity_type).items():
                        for question in question_list["questions"]:
                            sentence_question_pairs.append((idx, sentence, question, named_entity, relation, question_list["obj_types"]))
                            
        return sentence_question_pairs
    
    def classify_semantic_matching(self, sentence_question_pairs, batch_size):
        true_sentence_question_pairs = []
        
        for batches in self.get_batch(sentence_question_pairs, n=batch_size):
            sentences, questions = [], []
            
            for _, sentence, question, _, _, _ in batches:
                sentences.append(sentence)
                questions.append(question)
            
            results = self.sm_task.predict(sentence=sentences, question=questions)
            
            for idx, result in enumerate(results):
                if result:
                    sentence_idx = batches[idx][0]
                    entity = batches[idx][-3]
                    relation = batches[idx][-2]
                    obj_types = batches[idx][-1]
                    
                    true_sentence_question_pairs.append((
                        sentence_idx, 
                        sentences[idx], 
                        questions[idx], 
                        entity, 
                        relation, 
                        obj_types))
        
        return true_sentence_question_pairs
        
    def get_answer_from_sentence_question_pairs(self, sentence_question_pairs, batch_size, mrc_threshold):
        answers = []
        
        for batches in self.get_batch(sentence_question_pairs, n=batch_size):
            sentences, questions = [], []
            
            for _, sentence, question, _, _, _ in batches:
                sentences.append(sentence)
                questions.append(question)
                
            pred_answers = self.mrc_task.predict(sentence=sentences, question=questions)
            for idx, pred_answer in enumerate(pred_answers):
                answer = pred_answer["answer"]
                confidence_score = pred_answer["confidence_score"]
                
                if confidence_score < mrc_threshold:
                    continue
                
                sentence_idx = batches[idx][0]
                entity = batches[idx][-3]
                relation = batches[idx][-2]
                obj_types = batches[idx][-1]
                    
                answers.append((
                    sentence_idx, 
                    sentences[idx], 
                    questions[idx], 
                    entity,
                    relation,
                    answer, 
                    confidence_score, 
                    obj_types))
        
        return answers
        
    def get_triples(self, entities, answer):
        triples = []
        
        answers_per_sentences = {}
        for sentence_idx, sentence, _, subj, relation, obj, confidence_score, obj_types in answer:
            
            if not self.is_matched_answer_and_obj(sentence, entities[sentence_idx], subj, obj, obj_types):
                continue
            
            if sentence_idx not in answers_per_sentences:
                answers_per_sentences.setdefault(sentence_idx, {})
            
            so = f"{subj} {obj}"
            if so not in answers_per_sentences[sentence_idx]:
                answers_per_sentences[sentence_idx].setdefault(so, [])
                
            answers_per_sentences[sentence_idx][so].append({
                "relation": relation,
                "confidence_score": confidence_score
            })
            
        for idx, candidate_triple in answers_per_sentences.items():
            triples.append([])
            
            for so, relation_metadata in candidate_triple.items():
                subj, obj = so.split(" ")
                relation = max(relation_metadata, key=lambda x: x["confidence_score"])["relation"]
                
                triples[idx].append((subj, relation, obj))
                
        return triples
    
    def is_matched_answer_and_obj(self, sentence, entities, subj, answer, obj_types):
        for obj_info in entities:
            obj = sentence[obj_info["start_idx"]: obj_info["end_idx"] + 1]
            obj_type = obj_info["label"].lower()

            if not obj:
                continue
            
            if subj == obj:
                continue
            
            if (answer in obj or obj in answer) and obj_type in obj_types:
                return True
        
        return False
    
    def get_batch(self, data, n=1):
        l = len(data)
        for idx in range(0, l, n):
            yield data[idx: min(idx + n, l)]
    
    def cli(self):
        while True:
            print(self.run(sentence=input(">>> ")))    
    
    
from koria import KoRIA

koria = KoRIA(cfg_path="./", cfg_fn="koria.cfg")

sentence_list = ["홍길동의 아버지는 홍길춘이다."]
entities_list = koria.predict(task_name="ner", sentence=sentence_list)

for sentence, entities in zip(sentence_list, entities_list):
    for entity in entities:
        print(sentence[entity["start_idx"]: entity["end_idx"] + 1])



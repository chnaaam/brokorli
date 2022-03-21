from brokorli import BrokorliUnit

sm = BrokorliUnit.train(
    task_name="sm", 
    pretrained_model_name="monologg/koelectra-base-v3-discriminator",
    dataset_path="C://Users/chnam/KoIE-DataGene/data/sm",
    train_dataset_fn="train2.json",
    test_dataset_fn="test2.json",
    max_seq_len=512,
    optimizer="adamw", 
    epochs=3, 
    learning_rate="3e-5", 
    weight_decay=0.01, 
    train_batch_size=4,
    test_batch_size=4,
    train_num_workers=0,
    test_num_workers=0,
    model_hub_path="./model")

sm.train()

# from brokorli import BrokorliUnit

# ner = BrokorliUnit(task_name="ner")
# print(ner.predict(sentence="홍길동의 아버지는 홍길춘이다."))
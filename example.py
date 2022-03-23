from brokorli import Brokorli

extractor = Brokorli()

triples = extractor(
    sentence=[
        "홍길동의 아버지는 홍길춘이다.",
        "홍길동의 아버지는 홍길춘이다.",
        "홍길동의 아버지는 홍길춘이다.",
        "홍길동의 아버지는 홍길춘이다.",
        "홍길동의 아버지는 홍길춘이다.",
        "홍길동의 아버지는 홍길춘이다.",
    ],
    sm_batch_size=4, 
    sm_threshold=0.98, 
    mrc_batch_size=4, 
    mrc_threshold=0.98
)
print(triples)

# from brokorli import BrokorliUnit

# sm = BrokorliUnit(task_name="sm")
# sm.predict(sentence="홍길동의 아버지는 홍길춘이다.", question="홍길동의 어머니는?")
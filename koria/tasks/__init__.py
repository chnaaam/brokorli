from .named_entity_recognition import NER
from .machine_reading_comprehension import MRC
from .question_generation import QG

"""
TASK_LIST는 정의한 Task 클래스 들을 갖고 있습니다.
새로운 Task 클래스를 추가하고자 하는 경우, 아래 형식에 맞게 값을 추가해주세요.

Example
    TASK_LIST = {
        "task_name": {NEW_TASK_NAME},
    }
"""

TASK_LIST = {
    "ner": NER,
    "mrc": MRC,
    "qg": QG
}
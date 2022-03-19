from .sequence_labeling_model import SequenceLabelingModel
from .question_answering_model import QuestionAnsweringModel
from .sequence_classification_model import SequenceClassificationModel

"""
MODEL_LIST는 특정 Task를 위한 모델 클래스를 갖고 있습니다.
새로운 모델 클래스를 추가하고자 하는 경우, 아래 형식에 맞게 값을 추가해주세요.

Example
    MODEL_LIST = {
        "model name": {NEW_MODEL_NAME},
    }
"""

MODEL_LIST = {
    "ner": SequenceLabelingModel,
    "mrc": QuestionAnsweringModel,
    "sm": SequenceClassificationModel
}

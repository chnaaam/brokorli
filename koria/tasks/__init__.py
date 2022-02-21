from .semantic_role_labeling import SRL
from .named_entity_recognition import NER

TASK_LIST = {
    "srl": SRL,
    "ner": NER
}
import torch.optim as optim

from .rule_based_task import UserTemplate
from .named_entity_recognition import NER
from .machine_reading_comprehension import MRC
from .question_generation import QG
from .semantic_matching_classification import SM


TASK_LIST = {
    "ner": NER,
    "mrc": MRC,
    "qg": QG,
    "sm": SM
}

OPTIMIZER_LIST = {
    "adam": optim.Adam,
    "adamw": optim.AdamW
}

from .task_config import TaskConfig
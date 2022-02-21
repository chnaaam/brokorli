from .srl_dataset import SrlDataset
from .ner_dataset import NerDataset

DATASET_LIST = {
    "srl": SrlDataset,
    "ner": NerDataset
}
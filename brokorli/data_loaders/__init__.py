from .data import (
    NerData,
    MrcData,
    SmData
)

from .dataset import (
    NerDataset,
    MrcDataset,
    SmDataset
)

"""
DATA_LIST는 각 Task에 대한 Data 클래스를 갖고 있습니다.
여기서 Data 클래스는 데이터 셋을 불러오는 클래스 입니다.
만약, 새로운 Data 클래스를 만드는 경우, 아래 형식에 맞게 값을 추가해주세요.

Example
    DATA_LIST = {
        "new task 1": {NEW_DATA_CLASS_1},
        "new task 2": {NEW_DATA_CLASS_2}
    }
"""

DATA_LIST = {
    "ner": NerData,
    "mrc": MrcData,
    "sm": SmData
}

"""
DATASET_LIST 각 Task에 대한 Dataset 클래스를 갖고 있습니다.
여기서 Dataset 클래스는 데이터 셋을 불러오는 클래스 입니다.
만약, 새로운 Dataset 클래스를 만드는 경우, 아래 형식에 맞게 값을 추가해주세요.

Example
    DATASET_LIST = {
        "new task 1": {NEW_DATASET_CLASS_1},
        "new task 2": {NEW_DATASET_CLASS_2}
    }
"""

DATASET_LIST = {
    "ner": NerDataset,
    "mrc": MrcDataset,
    "sm": SmDataset
}

from .utils import load_data_loader
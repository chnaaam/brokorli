from .srl import SrlData, SrlDataset

"""
DATA_LIST는 각 Task에 대한 Data 클래스와 Dataset 클래스를 갖고 있습니다.
여기서 Data 클래스는 데이터 셋을 불러오는 클래스이고, Dataset은 torch.utils.data 내 Dataset 클래스를 상속 받아 사용하면 됩니다.
만약, 새로운 Data / Dataset 클래스를 만드는 경우, 아래 형식에 맞게 값을 추가해주세요.

Example
    DATA_LIST = {
        "new task": {
            "data": {NEW_DATA_CLASS},
            "dataset": {NEW_DATA_SET}
        }
    }
"""

DATA_LIST = {
    "srl": {
        "data": SrlData,
        "dataset": SrlDataset
    }
}
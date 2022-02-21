import os

from .dataset_loader import (
    SrlData,
    NerData,
    MrcData
)

from .dataset import (
    SrlDataset, 
    NerDataset,
    MrcDataset
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
    "srl": SrlData,
    "ner": NerData,
    "mrc": MrcData
}

DATASET_LIST = {
    "srl": SrlDataset,
    "ner": NerDataset,
    "mrc": MrcDataset
}

def load_data(task_cfg, task_name):
    # Load data when data parameter in configuration file is existed.
    # If data parameter is not existed, load train, valid, test dataset using configuration file.
    # Therefore, parameters must be added between dataset file names or specific dataset(train, valid, test) file names.
    
    # Check that train, valid, test file is existed
    train_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.train")
    valid_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.valid")
    test_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.test")
    
    # If three file is not existed, data is loaded
    train_data_list, valid_data_list, test_data_list = None, None, None
    
    if not os.path.exists(train_data_path) or not os.path.exists(valid_data_path) or not os.path.exists(test_data_path):        
        if "data" in task_cfg.dataset.__dict__:
            # Load dataset
            data = []
            
            # Multi dataset files
            if type(task_cfg.dataset.data) is list:
                for data_fn in task_cfg.dataset.data:
                    data += DATA_LIST[task_name](dataset_path=os.path.join(task_cfg.dataset.path, data_fn)).data
                    
            # Single dataset file
            else:
                data = DATA_LIST[task_name](dataset_path=os.path.join(task_cfg.dataset.path, task_cfg.dataset.data)).data
            
            import random
            import json
            
            random.shuffle(data)
            
            # Split data (Ratio - Train : Valid : Test = 8 : 1 : 1)
            len_data = len(data)
            len_train_data = int(len_data * 0.8)
            len_valid_data = int(len_data * 0.1)
            
            train_data_list = data[: len_train_data]
            valid_data_list = data[len_train_data: len_train_data + len_valid_data]
            test_data_list = data[len_train_data + len_valid_data : ]
            
            for path, data in [(train_data_path, train_data_list), (valid_data_path, valid_data_list), (test_data_path, test_data_list)]:
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(data, fp, ensure_ascii=False, indent=4)
            
            if not train_data_list and not valid_data_list and not test_data_list:
                raise ValueError("Dataset is empty")
            
        else:
            raise ValueError("The parameter of data is None")
    else:
        import json
        
        with open(train_data_path, "r", encoding="utf-8") as fp:
            train_data_list = json.load(fp)
                    
        with open(valid_data_path, "r", encoding="utf-8") as fp:
            valid_data_list = json.load(fp)
            
        with open(test_data_path, "r", encoding="utf-8") as fp:
            test_data_list = json.load(fp)
            
    return train_data_list, valid_data_list, test_data_list
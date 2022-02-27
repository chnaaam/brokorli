import os

from .data_list import (
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

def load_data(task_cfg, task_name, model_name, cache_dir):
    # Load data when data parameter in configuration file is existed.
    # If data parameter is not existed, load train, valid, test dataset using configuration file.
    # Therefore, parameters must be added between dataset file names or specific dataset(train, valid, test) file names.
    
    # If three file is not existed, data is loaded
    train_data_list, valid_data_list, test_data_list = None, None, None
    
    # Chat that cache file is existed
    cache_file_list = [fn for fn in os.listdir(cache_dir) if f"{task_name}-{model_name}-data.cache" in fn]
    
    if len(cache_file_list) == 3:
        return None, None, None
        
    # Check that train, valid, test file is existed
    train_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.train")
    valid_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.valid")
    test_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.test")
    
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
        
        elif "train" in task_cfg.dataset.__dict__ and "valid" in task_cfg.dataset.__dict__ and "test" in task_cfg.dataset.__dict__:
            train_data_list = DATA_LIST[task_name](dataset_path=os.path.join(task_cfg.dataset.path, task_cfg.dataset.train)).data
            valid_data_list = DATA_LIST[task_name](dataset_path=os.path.join(task_cfg.dataset.path, task_cfg.dataset.valid)).data
            test_data_list = DATA_LIST[task_name](dataset_path=os.path.join(task_cfg.dataset.path, task_cfg.dataset.test)).data
        
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
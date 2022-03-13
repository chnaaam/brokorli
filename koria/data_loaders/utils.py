import os
import json
import random
from torch.utils import data

from . import CACHE_FILE_FORMAT, DATA_LIST, DATASET_LIST

def load_data_loader(task_cfg, task_name, tokenizer, cache_dir, label_dir):
    datasets = {
        "train": {
            "data": None,
            "batch-size": task_cfg.parameters.train_batch_size,
            "num-workers": task_cfg.parameters.train_num_workers,
        },
        "valid": {
            "data": None,
            "batch-size": task_cfg.parameters.valid_batch_size,
            "num-workers": task_cfg.parameters.valid_num_workers,
        },
        "test": {
            "data": None,
            "batch-size": task_cfg.parameters.test_batch_size,
            "num-workers": task_cfg.parameters.test_num_workers,
        }
    }
    
    if len([fn for fn in os.listdir(cache_dir) if CACHE_FILE_FORMAT.format(task_name, task_cfg.model_name, task_cfg.parameters.max_seq_len) in fn]) == 3:
        return return_data_loader(task_cfg, task_name, tokenizer, cache_dir, label_dir, datasets)
    
    train_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.train")
    valid_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.valid")
    test_data_path = os.path.join(task_cfg.dataset.path, f"{task_name}.test")
    
    if is_existed_single_data_cfg(task_cfg.dataset) and not is_existed_train_valid_test_data_cfg(task_cfg.dataset):
        if not os.path.exists(train_data_path) or not os.path.exists(valid_data_path) or not os.path.exists(test_data_path):
            split_single_dataset_to_three(task_cfg, task_name, train_data_path, valid_data_path, test_data_path)        
            
        datasets["train"]["data"] = load_splitted_dataset(path=train_data_path)
        datasets["valid"]["data"] = load_splitted_dataset(path=valid_data_path)
        datasets["test"]["data"] = load_splitted_dataset(path=test_data_path)
        
    else:
        train_data_path = os.path.join(task_cfg.dataset.path, task_cfg.dataset.train_fn)
        valid_data_path = os.path.join(task_cfg.dataset.path, task_cfg.dataset.valid_fn)
        test_data_path = os.path.join(task_cfg.dataset.path, task_cfg.dataset.test_fn)

        datasets["train"]["data"] = DATA_LIST[task_name](dataset_path=train_data_path).data
        datasets["valid"]["data"] = DATA_LIST[task_name](dataset_path=valid_data_path).data
        datasets["test"]["data"] = DATA_LIST[task_name](dataset_path=test_data_path).data
    
    return return_data_loader(task_cfg, task_name, tokenizer, cache_dir, label_dir, datasets)
    
def is_existed_single_data_cfg(dataset):
    return "data_fn" in dataset.__dict__

def is_existed_train_valid_test_data_cfg(dataset):
    return "train_fn" in dataset.__dict__ and "valid_fn" in dataset.__dict__ and "test_fn" in dataset.__dict__

def split_single_dataset_to_three(task_cfg, task_name, train_data_path, valid_data_path, test_data_path):
    
    data = []
    fn_list = task_cfg.dataset.data_fn if type(task_cfg.dataset.data_fn) == list else [task_cfg.dataset.data_fn]
    
    for fn in fn_list:
        data += DATA_LIST[task_name](dataset_path=os.path.join(task_cfg.dataset.path, fn)).data

    random.shuffle(data)

    # Split data (Ratio - Train : Valid : Test = 8 : 1 : 1)
    len_data = len(data)
    len_train_data = int(len_data * 0.8)
    len_valid_data = int(len_data * 0.1)
    
    save_split_dataset(train_data_path, data[: len_train_data])
    save_split_dataset(valid_data_path, data[len_train_data: len_train_data + len_valid_data])
    save_split_dataset(test_data_path, data[len_train_data + len_valid_data : ])
    
def return_data_loader(task_cfg, task_name, tokenizer, cache_dir, label_dir, datasets):
    data_loaders = []
    
    for dataset_type, dataset in datasets.items():
        data_loaders.append(
            data.DataLoader(
                dataset=DATASET_LIST[task_name](
                    tokenizer=tokenizer,
                    task_name=task_name,
                    model_name=task_cfg.model_name,
                    data_list=dataset["data"], 
                    cache_dir=cache_dir,
                    label_dir=label_dir,
                    dataset_type=dataset_type,
                    max_seq_len=task_cfg.parameters.max_seq_len
                ),
                batch_size=dataset["batch-size"], 
                shuffle=task_cfg.parameters.dataset_shuffle, 
                pin_memory=task_cfg.parameters.pin_memory, 
                num_workers=dataset["num-workers"]
            )
        )
    
    return data_loaders

def load_splitted_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def save_split_dataset(path, data):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
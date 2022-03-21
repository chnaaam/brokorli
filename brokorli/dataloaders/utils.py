import os
import json
from torch.utils import data

from . import DATA_LIST, DATASET_LIST

def load_data_loader(
    task_cfg,
    dataset_path,
    train_dataset_fn,
    test_dataset_fn):
    
    datasets = {
        "train": {
            "data": None,
            "batch-size": task_cfg.train_batch_size,
            "num-workers": task_cfg.train_num_workers,
        },
        "test": {
            "data": None,
            "batch-size": task_cfg.test_batch_size,
            "num-workers": task_cfg.test_num_workers,
        }
    }
    
    if len([fn for fn in os.listdir(task_cfg.cache_dir) if "{}.{}".format(task_cfg.task_name, task_cfg.pretrained_model_name.split("/")[1]) in fn]) == 2:
        return return_data_loader(task_cfg, datasets)
    
    train_data_path = os.path.join(dataset_path, train_dataset_fn)
    test_data_path = os.path.join(dataset_path, test_dataset_fn)

    datasets["train"]["data"] = DATA_LIST[task_cfg.task_name](dataset_path=train_data_path).data
    datasets["test"]["data"] = DATA_LIST[task_cfg.task_name](dataset_path=test_data_path).data
    
    return return_data_loader(task_cfg, datasets)
    
def return_data_loader(task_cfg, datasets):
    data_loaders = []
    
    for dataset_type, dataset in datasets.items():
        data_loaders.append(
            data.DataLoader(
                dataset=DATASET_LIST[task_cfg.task_name](
                    tokenizer=task_cfg.tokenizer,
                    task_name=task_cfg.task_name,
                    model_name=task_cfg.pretrained_model_name.split("/")[1],
                    data_list=dataset["data"], 
                    cache_dir=task_cfg.cache_dir,
                    label_dir=task_cfg.label_hub_path,
                    dataset_type=dataset_type,
                    max_seq_len=task_cfg.max_seq_len
                ),
                batch_size=dataset["batch-size"], 
                shuffle=task_cfg.dataset_shuffle, 
                pin_memory=task_cfg.pin_memory, 
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
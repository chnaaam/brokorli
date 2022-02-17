import os
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

from .config import get_data_gene_config
from .utils import *
from .data_loaders import DATA_LIST
from .tasks import TASK_LIST
from .models import LAYER_LIST
from .tokenizers import TOKENIZER_LIST
from .functions import CRITERION_LIST, OPTIMIZER_LIST, SCHEDULER_LIST


class DataGene:
    
    def __init__(self, cfg_path, cfg_fn):
        cfg_full_path = os.path.join(cfg_path, cfg_fn)
        
        if not is_existed_file(cfg_full_path):
            raise OSError("Configuration directory or file is not existed")
        
        self.cfg = get_data_gene_config(cfg_path=cfg_path, cfg_fn=cfg_fn)
        
        for path in vars(self.cfg.path).values():
            make_dir(os.path.join(self.cfg.path.root, path))
            
    def train(self, task_name):
        
        if task_name == "srl":
            task_cfg = self.cfg.tasks.srl
        else:
            raise NotImplementedError()
    
        # Build model
        layer_list = {layer_name: LAYER_LIST[layer_name] for layer_name in task_cfg.model.split("-")}
        
        # Load Tokenizer
        tokenizer = TOKENIZER_LIST[task_cfg.tokenizer]
        
        # Load cache dataset or build dataset for training
        cache_dir = self.cfg.path.cache
        train_cache_path = os.path.join(cache_dir, f"{task_name}-train-data.cache")
        valid_cache_path = os.path.join(cache_dir, f"{task_name}-valid-data.cache")
        test_cache_path = os.path.join(cache_dir, f"{task_name}-test-data.cache")
        
        # Build dataset
        train_data_list, valid_data_list, test_data_list = None, None, None
        
        if not os.path.exists(train_cache_path) or not os.path.exists(valid_cache_path) or not os.path.exists(test_cache_path):
            train_data_list, valid_data_list, test_data_list = self.load_dataset(task_name=task_name, task_cfg=task_cfg)
        
            if not train_data_list and not valid_data_list and not test_data_list:
                raise ValueError("Dataset is empty")
        
        train_dataset = DATA_LIST[task_name]["dataset"](
            tokenizer=tokenizer,
            data_list=train_data_list, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
            vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
            dataset_type="train",
            max_seq_len=self.cfg.parameters.max_seq_len)
        
        valid_dataset = DATA_LIST[task_name]["dataset"](
            tokenizer=tokenizer,
            data_list=valid_data_list, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
            vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
            dataset_type="valid",
            max_seq_len=self.cfg.parameters.max_seq_len)
        
        test_dataset = DATA_LIST[task_name]["dataset"](
            tokenizer=tokenizer,
            data_list=test_data_list, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
            vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
            dataset_type="test",
            max_seq_len=self.cfg.parameters.max_seq_len)
        
        # Make environment for selected task
        
        task = TASK_LIST[task_name](
            parameter_cfg=self.cfg.parameters,
            layer_list=layer_list,
            tokenizer=tokenizer,
            data_loader={
                "train": data.DataLoader(train_dataset, batch_size=self.cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.parameters.num_workers),
                "valid": data.DataLoader(valid_dataset, batch_size=self.cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.parameters.num_workers),
                "test": data.DataLoader(test_dataset, batch_size=self.cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.parameters.num_workers),
            },
            optimizer=OPTIMIZER_LIST[self.cfg.parameters.optimizer],
            criterion=CRITERION_LIST[self.cfg.parameters.criterion],
            scheduler=SCHEDULER_LIST[self.cfg.parameters.scheduler],
            use_cuda=self.cfg.use_cuda,
            model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
            
            # Optional Parameters
            vocab_size=len(train_dataset.tokenizer),
            label_size=len(train_dataset.l2i) if "l2i" in vars(train_dataset).keys() else None,
            token_pad_id=tokenizer.pad_token_id,
            l2i=train_dataset.l2i if "l2i" in vars(train_dataset).keys() else None,
            i2l=train_dataset.i2l if "i2l" in vars(train_dataset).keys() else None,
            special_label_tokens=train_dataset.SPECIAL_LABEL_TOKENS if "SPECIAL_LABEL_TOKENS" in vars(train_dataset).keys() else None
        )
        
        task.train()
    
    def load_dataset(self, task_name, task_cfg):
        
        # Build dataset
        if "data" in task_cfg.dataset.__dict__:
            data = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.data).data
            
            import random
            random.shuffle(data)
            
            # Split data (Ratio - Train : Valid : Test = 8 : 1 : 1)
            len_data = len(data)
            len_train_data = int(len_data * 0.8)
            len_valid_data = int(len_data * 0.1)
            
            train_data_list = data[: len_train_data]
            valid_data_list = data[len_train_data: len_train_data + len_valid_data]
            test_data_list = data[len_train_data + len_valid_data : ]
        else:
            train_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.train).data
            valid_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.valid).data
            test_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.test).data
        
        return train_data_list, valid_data_list, test_data_list
    
    def predict(self):
        pass
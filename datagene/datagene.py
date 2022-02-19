import os
import torch.utils.data as data

from . import MODEL_NAME_LIST
from .config import get_data_gene_config
from .utils import *
from .data_loaders import DATA_LIST
from .tasks import TASK_LIST
from .models import MODEL_LIST
from .tokenizers import TOKENIZER_LIST
from .functions import CRITERION_LIST, OPTIMIZER_LIST, SCHEDULER_LIST


class DataGene:
    
    def __init__(self, cfg_path, cfg_fn):
        
        if not is_existed_file(os.path.join(cfg_path, cfg_fn)):
            raise OSError("Configuration directory or file is not existed")
        
        self.cfg = get_data_gene_config(cfg_path=cfg_path, cfg_fn=cfg_fn)
        
        for path in vars(self.cfg.path).values():
            make_dir(os.path.join(self.cfg.path.root, path))
            
    def train(self, task_name):
        
        if task_name == "srl":
            task_cfg = self.cfg.tasks.srl
        else:
            raise NotImplementedError()
            
        # Load Tokenizer
        tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name])
        
        # Load data
        train_data_list, valid_data_list, test_data_list = self.load_dataset(task_name=task_name, task_cfg=task_cfg)
        
        # Load dataset
        train_dataset = DATA_LIST[task_name]["dataset"](
            tokenizer=tokenizer,
            model_name=task_cfg.model_name,
            data_list=train_data_list, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
            vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
            dataset_type="train",
            max_seq_len=self.cfg.parameters.max_seq_len)
        
        valid_dataset = DATA_LIST[task_name]["dataset"](
            tokenizer=tokenizer,
            model_name=task_cfg.model_name,
            data_list=valid_data_list, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
            vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
            dataset_type="valid",
            max_seq_len=self.cfg.parameters.max_seq_len)
        
        test_dataset = DATA_LIST[task_name]["dataset"](
            tokenizer=tokenizer,
            model_name=task_cfg.model_name,
            data_list=test_data_list, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
            vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
            dataset_type="test",
            max_seq_len=self.cfg.parameters.max_seq_len)
        
        # Make environment for selected task
        task = TASK_LIST[task_name](
            
            # Configuration for training
            parameter_cfg=self.cfg.parameters,
            
            # Selected LM Model
            # - model name = {}
            model_name=MODEL_NAME_LIST[task_cfg.model_name],
            model_type=task_cfg.model_type,
            
            # Tokenizer
            tokenizer=tokenizer,
            
            # Train, valid, test data Loader
            data_loader={
                "train": data.DataLoader(train_dataset, batch_size=self.cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.parameters.num_workers),
                "valid": data.DataLoader(valid_dataset, batch_size=self.cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.parameters.num_workers),
                "test": data.DataLoader(test_dataset, batch_size=self.cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=self.cfg.parameters.num_workers),
            },
            
            # Optimizer, loss function, scheduler
            optimizer=OPTIMIZER_LIST[self.cfg.parameters.optimizer],
            criterion=CRITERION_LIST[self.cfg.parameters.criterion],
            scheduler=SCHEDULER_LIST[self.cfg.parameters.scheduler],
            
            # Use GPU or not
            use_cuda=self.cfg.use_cuda,
            
            # The model hub is a directory where models are stored when model training is over.
            model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
            
            # Optional Parameters
            # If a special token is added, the input size of the model is adjusted.
            vocab_size=len(train_dataset.tokenizer),
            
            # Label Size
            label_size=len(train_dataset.l2i) if "l2i" in vars(train_dataset).keys() else None,
            
            # Token index in tokenizer and label-index pair
            token_pad_id=tokenizer.pad_token_id,
            l2i=train_dataset.l2i if "l2i" in vars(train_dataset).keys() else None,
            i2l=train_dataset.i2l if "i2l" in vars(train_dataset).keys() else None,
            
            # Special token list for decoding sequence labeling outputs
            special_label_tokens=train_dataset.SPECIAL_LABEL_TOKENS if "SPECIAL_LABEL_TOKENS" in vars(train_dataset).keys() else None
        )
        
        # Train
        task.train()
    
    def load_dataset(self, task_name, task_cfg):
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
                data = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.data).data
                
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
                train_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.train).data
                valid_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.valid).data
                test_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=task_cfg.dataset.test).data
        else:
            train_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=f"{task_name}-{task_cfg.model_name}.train").data
            valid_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=f"{task_name}-{task_cfg.model_name}.valid").data
            test_data_list = DATA_LIST[task_name]["data"](dataset_dir=task_cfg.dataset.path, dataset_fn=f"{task_name}-{task_cfg.model_name}.test").data
        
        return train_data_list, valid_data_list, test_data_list
    
    def predict(self):
        pass
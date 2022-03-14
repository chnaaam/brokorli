import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import pickle
import logging

logger = logging.getLogger("koria")

from . import MODEL_NAME_LIST
from .config import get_config
from .utils import *
from .data_loaders import load_data_loader
from .tasks import TASK_LIST, TaskConfig
from .tokenizers import TOKENIZER_LIST
from .workflow import Workflow
from .dashboard import Dashboard


class KoRIA:
    
    def __init__(self, cfg_path, cfg_fn, run_type):
        self.run_type = run_type
        
        if not is_existed_file(os.path.join(cfg_path, cfg_fn)):
            raise OSError("Configuration directory or file is not existed")
        
        logger.info("Load configuration file")
        self.cfg = get_config(cfg_path=cfg_path, cfg_fn=cfg_fn)
        
        for path in vars(self.cfg.path).values():
            make_dir(os.path.join(self.cfg.path.root, path))
        
    def train(self, task_name):
        
        if task_name not in ["ner", "mrc", "sm"]:
            raise ValueError("{task_name} task is not supported for training")
        
        logger.info(f"Task : {task_name}")
        
        task_cfg = get_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
        
        if not task_cfg:
            raise ValueError(f"The task is not defined. Define the {task_name} task.")
        
        # Load Tokenizer
        logger.info(f"Model name : {task_cfg.model_name}")
        logger.info(f"Model type : {task_cfg.model_type}")
        logger.info(f"Tokenizer name : {task_cfg.model_name}")
        
        tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name], use_fast=True)
        
        # Load data
        logger.info(f"Load data")
        
        train_data_loader, valid_data_loader, test_data_loader = load_data_loader(
            task_cfg=task_cfg, 
            task_name=task_name, 
            tokenizer=tokenizer, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache), 
            label_dir=os.path.join(self.cfg.path.root, self.cfg.path.label)
        )
        
        # Make environment for selected task
        logger.info(f"Make environment for {task_name} task")
        
        task_config = TaskConfig(
            task_name=task_name,
            task_cfg=task_cfg,
            model_name=MODEL_NAME_LIST[task_cfg.model_name], 
            tokenizer=tokenizer,
            train_mode=True if self.run_type == "train" else False,
            train_data_loader=train_data_loader, 
            valid_data_loader=valid_data_loader, 
            test_data_loader=test_data_loader, 
            model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
            label_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.label),
            use_cuda=self.cfg.use_cuda,
            use_fp16=self.cfg.use_fp16
        )
        
        task = TASK_LIST[task_name](task_config=task_config)
        task.train()
    
    def predict(self, task_name, **parameters):
        task_cfg = get_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
        
        if not task_cfg:
            raise ValueError(f"The task is not defined. Define the {task_name} task.")
        
        # Load Tokenizer
        logger.info(f"Model name : {task_cfg.model_name}")
        logger.info(f"Model type : {task_cfg.model_type}")
        logger.info(f"Tokenizer name : {task_cfg.model_name}")
        
        tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name], use_fast=True)
        
        # Make environment for selected task
        logger.info(f"Make environment for {task_name} task")
        
        task_config = TaskConfig(
            task_name=task_name,
            task_cfg=task_cfg,
            model_name=MODEL_NAME_LIST[task_cfg.model_name], 
            tokenizer=tokenizer,
            train_mode=True if self.run_type == "train" else False,
            train_data_loader=None, 
            valid_data_loader=None, 
            test_data_loader=None, 
            model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
            label_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.label),
            use_cuda=self.cfg.use_cuda,
            use_fp16=self.cfg.use_fp16
        )
        
        task = TASK_LIST[task_name](task_config=task_config)
        
        if task_name == "ner":
            res = task.predict(sentence=parameters["sentence"])
        elif task_name == "mrc":
            res = task.predict(sentence=parameters["sentence"], question=parameters["question"])
        elif task_name == "sm":
            res = task.predict(sentence=parameters["sentence"], question=parameters["question"])
            
        print(res)
        
    def demo(self):
        tasks = {}
        
        for task_name in TASK_LIST.keys():
            task_cfg = get_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
            
            if task_name in ["ner", "mrc", "sm"]:
                tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name], use_fast=True)
                task_config = TaskConfig(
                    task_name=task_name,
                    task_cfg=task_cfg,
                    model_name=MODEL_NAME_LIST[task_cfg.model_name], 
                    tokenizer=tokenizer,
                    train_mode=True if self.run_type == "train" else False,
                    train_data_loader=None, 
                    valid_data_loader=None, 
                    test_data_loader=None, 
                    model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
                    label_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.label),
                    use_cuda=self.cfg.use_cuda,
                    use_fp16=self.cfg.use_fp16
                )
                
                tasks.setdefault(task_name, TASK_LIST[task_name](task_config=task_config))
            else:
                task = TASK_LIST[task_name](
                    cfg=task_cfg,
                    template_dir=self.cfg.path.template
                )
                
                tasks.setdefault(task_name, task)
            
        dashboard = Dashboard(tasks=tasks)
        dashboard.run()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import pickle
import logging

logger = logging.getLogger("koria")

import torch.utils.data as data

from . import MODEL_NAME_LIST, OPTIMIZER_LIST
from .config import get_data_gene_config
from .utils import *
from .data_loaders import load_data, DATASET_LIST
from .tasks import TASK_LIST
from .tokenizers import TOKENIZER_LIST, SPECIAL_TOKEN_LIST
from .workflow import Workflow

class KoRIA:
    
    def __init__(self, cfg_path, cfg_fn):
        
        if not is_existed_file(os.path.join(cfg_path, cfg_fn)):
            raise OSError("Configuration directory or file is not existed")
        
        logger.info("Load configuration file")
        self.cfg = get_data_gene_config(cfg_path=cfg_path, cfg_fn=cfg_fn)
        
        for path in vars(self.cfg.path).values():
            make_dir(os.path.join(self.cfg.path.root, path))
            
    def train(self, task_name):
        
        if task_name in ["qg"]:
            raise ValueError("{task_name} task is not supported for training")
        
        logger.info(f"Task : {task_name}")
        
        task_cfg = get_data_gene_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
        
        if not task_cfg:
            raise ValueError(f"The task is not defined. Define the {task_name} task.")
        
        # Load Tokenizer
        logger.info(f"Model name : {task_cfg.model_name}")
        logger.info(f"Model type : {task_cfg.model_type}")
        logger.info(f"Tokenizer name : {task_cfg.model_name}")
        
        tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name], use_fast=True)
        
        # Add special tokens in tokenizer
        if task_name in SPECIAL_TOKEN_LIST:
            tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKEN_LIST[task_name].values())})
        
        # Load data
        logger.info(f"Load data")
        
        train_data_list, valid_data_list, test_data_list = load_data(
            task_cfg=task_cfg, 
            task_name=task_name, 
            model_name=task_cfg.model_name, 
            cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache))
        
        # Load dataset
        dataset_list = []
        
        for dataset_type, data_list in [("train", train_data_list), ("valid", valid_data_list), ("test", test_data_list)]:
            dataset_list.append(DATASET_LIST[task_name](
                tokenizer=tokenizer,
                task_name=task_name,
                special_tokens=SPECIAL_TOKEN_LIST[task_name] if task_name in SPECIAL_TOKEN_LIST else None,
                model_name=task_cfg.model_name,
                data_list=data_list, 
                cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
                vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
                dataset_type=dataset_type,
                max_seq_len=task_cfg.parameters.max_seq_len))
        
        train_dataset, valid_dataset, test_dataset = dataset_list
        
        # Make environment for selected task
        logger.info(f"Make environment for {task_name} task")
        
        task = TASK_LIST[task_name](
            
            # Configuration for selected task
            cfg=task_cfg,
            
            # Selected LM Model
            model_name=MODEL_NAME_LIST[task_cfg.model_name],
            model_type=task_cfg.model_type,
            
            # Tokenizer
            tokenizer=tokenizer,
            
            # Train, valid, test data Loader
            data_loader={
                "train": data.DataLoader(train_dataset, batch_size=task_cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=task_cfg.parameters.num_workers),
                "valid": data.DataLoader(valid_dataset, batch_size=task_cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=task_cfg.parameters.num_workers),
                "test": data.DataLoader(test_dataset, batch_size=task_cfg.parameters.batch_size, shuffle=True, pin_memory=True, num_workers=task_cfg.parameters.num_workers),
            },
            
            # Optimizer, loss function, scheduler
            optimizer=OPTIMIZER_LIST[task_cfg.parameters.optimizer],
            
            # Use GPU or not
            use_cuda=self.cfg.use_cuda,
            
            # Use fp16 training
            use_fp16=self.cfg.use_fp16,
            
            # The model hub is a directory where models are stored when model training is over.
            model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
            
            # Optional Parameters
            # If a special token is added, the input size of the model is adjusted.
            vocab_size=len(train_dataset.tokenizer),
            
            # Label Size
            label_size=len(train_dataset.l2i) if "l2i" in vars(train_dataset).keys() else None,
            
            # Token index in tokenizer and label-index pair
            l2i=train_dataset.l2i if "l2i" in vars(train_dataset).keys() else None,
            i2l=train_dataset.i2l if "i2l" in vars(train_dataset).keys() else None,
            
            # Special token list for decoding sequence labeling outputs
            special_label_tokens=train_dataset.SPECIAL_LABEL_TOKENS if "SPECIAL_LABEL_TOKENS" in vars(train_dataset).keys() else None
        )
        
        # Train
        task.train()
        # task.valid()
    
    def predict(self, task_name):
        task_cfg = get_data_gene_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
        
        if not task_cfg:
            raise ValueError(f"The task is not defined. Define the {task_name} task.")
        
        if task_name not in ["qg"]:
            tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name])
            
            # Add special tokens in tokenizer
            if task_name in SPECIAL_TOKEN_LIST:
                tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKEN_LIST[task_name].values())})
            
            l2i, i2l = None, None
            if task_name in ["ner"]:
                with open(os.path.join(self.cfg.path.root, self.cfg.path.vocab, f"{task_name}.label"), "rb") as fp:
                    data = pickle.load(fp)

                if "l2i" not in data.keys():
                    raise KeyError("Invalid label file. Please check label file and run it again")

                l2i = data["l2i"]
                i2l = {i: l for l, i in l2i.items()}
            
            task = TASK_LIST[task_name](
                
                # Configuration for training
                cfg=task_cfg,
                
                # Selected LM Model
                model_name=MODEL_NAME_LIST[task_cfg.model_name],
                model_type=task_cfg.model_type,
                
                # Tokenizer
                tokenizer=tokenizer,
                
                # Use GPU or not
                use_cuda=self.cfg.use_cuda,
                
                # The model hub is a directory where models are stored when model training is over.
                model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
                
                # Optional Parameters
                # If a special token is added, the input size of the model is adjusted.
                vocab_size=len(tokenizer),
                
                # Label Size
                label_size=len(l2i) if l2i else None,
                
                # Token index in tokenizer and label-index pair
                token_pad_id=tokenizer.pad_token_id,
                l2i=l2i if l2i else None,
                i2l=i2l if i2l else None,
            )
            
            print(
                task.predict(
                    # sentence="아이유(IU, 본명: 이지은, 李知恩[1], 1993년 5월 16일~)는 대한민국의 가수이자 배우이다. 배우로 활동할 때는 본명을 사용한다. '아이유(IU)'라는 예명은 'I'와 'You'를 합친 합성어로 '너와 내가 음악으로 하나가 된다.'라는 의미이다.", 
                    sentence="홍길동의 아버지는 홍길춘이다.",
                    question="홍길동 어머니의 이름은?"
                )
            )
            # task.predict(sentence="홍길동의 아버지는 홍길춘이다.")
            
        else:
            task = TASK_LIST[task_name](
                
                # Configuration
                cfg=task_cfg,
                
                # rule root directory
                rule_dir=self.cfg.path.rules
            )
            
            task.predict(entity="조세호", type="ps")
        
    def cli(self):
        tasks = {}
        
        for task_name in TASK_LIST.keys():
            task_cfg = get_data_gene_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
            
            if task_name not in ["qg"]:
                tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name])
            
                # Add special tokens in tokenizer
                if task_name in SPECIAL_TOKEN_LIST:
                    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKEN_LIST[task_name].values())})
                
                l2i, i2l = None, None
                if task_name in ["ner"]:
                    with open(os.path.join(self.cfg.path.root, self.cfg.path.vocab, f"{task_name}.label"), "rb") as fp:
                        data = pickle.load(fp)

                    if "l2i" not in data.keys():
                        raise KeyError("Invalid label file. Please check label file and run it again")

                    l2i = data["l2i"]
                    i2l = {i: l for l, i in l2i.items()}
            
                task = TASK_LIST[task_name](
                    
                    # Configuration for training
                    cfg=task_cfg,
                    
                    # Selected LM Model
                    model_name=MODEL_NAME_LIST[task_cfg.model_name],
                    model_type=task_cfg.model_type,
                    
                    # Tokenizer
                    tokenizer=tokenizer,
                    
                    # Use GPU or not
                    use_cuda=self.cfg.use_cuda,
                    
                    # The model hub is a directory where models are stored when model training is over.
                    model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
                    
                    # Optional Parameters
                    # If a special token is added, the input size of the model is adjusted.
                    vocab_size=len(tokenizer),
                    
                    # Label Size
                    label_size=len(l2i) if l2i else None,
                    
                    # Token index in tokenizer and label-index pair
                    token_pad_id=tokenizer.pad_token_id,
                    l2i=l2i if l2i else None,
                    i2l=i2l if i2l else None,
                )
            else:
                task = TASK_LIST[task_name](
                    # Configuration
                    cfg=task_cfg,
                    
                    # rule root directory
                    rule_dir=self.cfg.path.rules
                )
                
            tasks.setdefault(task_name, task)
            
        workflow = Workflow(tasks=tasks)
        workflow.cli()
        
import os
import torch.utils.data as data

from . import MODEL_NAME_LIST
from .config import get_data_gene_config
from .utils import *
from .data_loaders import load_data, DATASET_LIST
from .tasks import TASK_LIST
from .tokenizers import TOKENIZER_LIST, SPECIAL_TOKEN_LIST
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
        
        # Add special tokens in tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKEN_LIST[task_name].values())})
        
        # Load data
        train_data_list, valid_data_list, test_data_list = load_data(task_cfg=task_cfg, task_name=task_name)
        
        # Load dataset
        dataset_list = []
        for dataset_type, data_list in [("train", train_data_list), ("valid", valid_data_list), ("test", test_data_list)]:
            dataset_list.append(DATASET_LIST[task_name](
                tokenizer=tokenizer,
                special_tokens=SPECIAL_TOKEN_LIST[task_name],
                model_name=task_cfg.model_name,
                data_list=data_list, 
                cache_dir=os.path.join(self.cfg.path.root, self.cfg.path.cache),
                vocab_dir=os.path.join(self.cfg.path.root, self.cfg.path.vocab),
                dataset_type=dataset_type,
                max_seq_len=self.cfg.parameters.max_seq_len))
        
        train_dataset, valid_dataset, test_dataset = dataset_list
        
        # Make environment for selected task
        task = TASK_LIST[task_name](
            
            # Configuration for training
            parameter_cfg=self.cfg.parameters,
            
            # Selected LM Model
            # Model List
            # "bert": "monologg/kobert",
            # "electra": "monologg/koelectra-base-v3-discriminator",
            # "charelectra": "monologg/kocharelectra-base-discriminator",
            # "roberta": "klue/roberta-base"
            # 
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
    
    def predict(self):
        pass
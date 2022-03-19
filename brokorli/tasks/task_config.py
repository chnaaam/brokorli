import os
import brokorli

from pathlib import Path
from appdirs import user_cache_dir
from transformers import AutoTokenizer


from . import OPTIMIZER_LIST

PRETRAINED_MODEL_NAME_LIST = {
    "ner": "brokorli/brokorli_ner",
    "mrc": "brokorli/brokorli_mrc",
    "sm": "brokorli/brokorli_sm"
}

class TaskConfig:
    def __init__(
        self, 
        task_name, 
        pretrained_model_name="",
        max_seq_len=512,
        optimizer="adamw",
        epochs=3,
        learning_rate=1e-5,
        weight_decay=0,
        train_batch_size=4,
        test_batch_size=4,
        train_num_workers=0,
        test_num_workers=0,
        model_hub_path="",
        dataset_shuffle=True,
        pin_memory=True,
        use_cuda=False, 
        use_fp16=False):
        
        self.task_name = task_name
        
        if pretrained_model_name:
            self.pretrained_model_name = pretrained_model_name
        else:
            if self.task_name in PRETRAINED_MODEL_NAME_LIST:
                self.pretrained_model_name = PRETRAINED_MODEL_NAME_LIST[self.task_name]
                self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, use_fast=True)
                
                self.train_data_loader, self.test_data_loader = None, None
        
                self.max_seq_len = max_seq_len
                self.optimizer = OPTIMIZER_LIST[optimizer] if optimizer else None
                
                self.epochs = epochs
                self.learning_rate = learning_rate
                self.weight_decay = weight_decay
                
                self.train_batch_size = train_batch_size
                self.test_batch_size = test_batch_size
                
                self.train_num_workers = train_num_workers
                self.test_num_workers = test_num_workers
                
                self.model_hub_path = model_hub_path
                
                self.dataset_shuffle = dataset_shuffle
                self.pin_memory = pin_memory
                
                self.use_cuda = use_cuda
                self.use_fp16 = use_fp16
                
                self.cache_dir = user_cache_dir("brokorli")
                
                if not os.path.isdir(self.cache_dir):
                    Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            else:
                self.pretrained_model_name = ""
                self.template_dir = os.path.join(os.path.dirname(brokorli.__file__), "templates")
        
        self.label_hub_path = os.path.join(os.path.dirname(brokorli.__file__), "label")
        
    def set_data_loader(self, train_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
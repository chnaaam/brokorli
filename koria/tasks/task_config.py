from . import OPTIMIZER_LIST

class TaskConfig:
    def __init__(
        self, 
        
        task_name,
        task_cfg,
        model_name, 
        tokenizer,
        
        train_mode = False,
        
        train_data_loader=None, 
        valid_data_loader=None, 
        test_data_loader=None, 
                
        model_hub_path="",
        label_hub_path="",
        
        use_cuda=False, 
        use_fp16=False):
        
        self.train_mode = train_mode
        self.task_name = task_name
        self.model_name = model_name
        self.model_type = task_cfg.model_type
        self.tokenizer = tokenizer
        
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        
        self.optimizer = OPTIMIZER_LIST[task_cfg.parameters.optimizer]
        
        self.epochs = task_cfg.parameters.epochs
        self.learning_rate = task_cfg.parameters.learning_rate
        self.weight_decay = task_cfg.parameters.weight_decay
        
        self.train_batch_size = task_cfg.parameters.train_batch_size
        self.valid_batch_size = task_cfg.parameters.valid_batch_size
        self.test_batch_size = task_cfg.parameters.test_batch_size
        
        self.train_num_workers = task_cfg.parameters.train_num_workers
        self.valid_num_workers = task_cfg.parameters.valid_num_workers
        self.test_num_workers = task_cfg.parameters.test_num_workers
        
        self.max_seq_len = task_cfg.parameters.max_seq_len
        
        self.model_hub_path = model_hub_path
        self.label_hub_path = label_hub_path
        
        self.use_cuda = use_cuda
        self.use_fp16 = use_fp16
    
class TaskConfig:
    def __init__(
        self, 
        
        model_name, 
        model_type, 
        tokenizer,
        
        # Training parameters
        train_dataset=None, 
        valid_dataset=None, 
        test_dataset=None, 
        
        optimizer=None,
        
        epochs=0,
        learning_rate=0.0001, 
        weight_decay=0,
        train_batch_size=1,
        eval_batch_size=1,
        dataloader_num_workers=0,
        max_seq_len=256, 
        
        model_hub_path="",
        label_hub_path="",
        
        use_gpu=False, 
        use_fp16=False):
        
        self.model_name = model_name
        self.model_type = model_type
        self.tokenizer = tokenizer
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
        self.optimizer = optimizer
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.max_seq_len = max_seq_len
        
        self.model_hub_path = model_hub_path
        self.label_hub_path = label_hub_path
        
        self.use_gpu = use_gpu
        self.use_fp16 = use_fp16
    
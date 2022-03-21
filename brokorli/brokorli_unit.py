import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from .utils import *
from .dataloaders import load_data_loader
from .tasks import TASK_LIST, TaskConfig


class BrokorliUnit:
    def __init__(
        self, 
        task_name,
        run_type="predict",
        pretrained_model_name=None,
        dataset_path=None,
        train_dataset_fn=None,
        test_dataset_fn=None,
        max_seq_len=512,
        optimizer="adamw", 
        epochs=3, 
        learning_rate="3e-5", 
        weight_decay=0.01, 
        train_batch_size=4,
        test_batch_size=4,
        train_num_workers=0,
        test_num_workers=0,
        model_hub_path="./model",
        use_cuda=True, 
        use_fp16=True
    ):
        self.task_name = task_name
        
        if run_type == "predict":
            self.task_config = TaskConfig(
                run_type=run_type,
                task_name=task_name,
                max_seq_len=max_seq_len,
                use_cuda=use_cuda,
                use_fp16=use_fp16
            )
            
            self.task = TASK_LIST[task_name](task_config=self.task_config)
        else:
            if self.task_name not in ["ner", "mrc", "sm"]:
                raise ValueError("{task_name} task is not supported for training")
            
            self.task_config = TaskConfig(
                run_type=run_type,
                task_name=task_name,
                pretrained_model_name=pretrained_model_name,
                max_seq_len=max_seq_len,
                optimizer=optimizer,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                train_batch_size=train_batch_size,
                test_batch_size=test_batch_size,
                train_num_workers=train_num_workers,
                test_num_workers=test_num_workers,
                model_hub_path=model_hub_path,
                use_cuda=use_cuda,
                use_fp16=use_fp16
            )
            
            train_data_loader, test_data_loader = load_data_loader(
                task_cfg=self.task_config,
                dataset_path=dataset_path,
                train_dataset_fn=train_dataset_fn,
                test_dataset_fn=test_dataset_fn,
            )
            
            self.task_config.set_data_loader(train_data_loader, test_data_loader)
            
            self.task = TASK_LIST[task_name](task_config=self.task_config)
            self.task.train()
    
    def predict(self, **parameters):
        if self.task_name == "ner":
            res = self.task.predict(sentence=parameters["sentence"])
            
        elif self.task_name == "mrc":
            res = self.task.predict(
                sentence=parameters["sentence"], 
                question=parameters["question"])
            
        elif self.task_name == "sm":
            res = self.task.predict(
                sentence=parameters["sentence"], 
                question=parameters["question"])
            
        elif self.task_name == "qg":
            res = self.task.predict(
                entity=parameters["entity"], 
                entity_type=parameters["entity_type"], 
                with_entity_marker=parameters["with_entity_marker"] if "with_entity_marker" in parameters else "")
            
        else:
            raise NotImplementedError()
        
        return res
    
    @classmethod
    def train(
        cls,
        task_name,
        pretrained_model_name,
        dataset_path,
        train_dataset_fn,
        test_dataset_fn,
        max_seq_len=512,
        optimizer="adamw", 
        epochs=3, 
        learning_rate="3e-5", 
        weight_decay=0.01, 
        train_batch_size=4,
        test_batch_size=4,
        train_num_workers=0,
        test_num_workers=0,
        model_hub_path="./model"):
        
        return cls(
            task_name=task_name,
            run_type="train",
            pretrained_model_name=pretrained_model_name,
            dataset_path=dataset_path,
            train_dataset_fn=train_dataset_fn,
            test_dataset_fn=test_dataset_fn,
            max_seq_len=max_seq_len,
            optimizer=optimizer,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_num_workers=train_num_workers,
            test_num_workers=test_num_workers,
            model_hub_path=model_hub_path,
        )
    
        
            
    # def task_demo(self):
    #     tasks = {}
        
    #     for task_name in TASK_LIST.keys():
    #         task_cfg = get_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
            
    #         if task_name in ["ner", "mrc", "sm"]:
    #             tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name], use_fast=True)
    #             task_config = TaskConfig(
    #                 task_name=task_name,
    #                 task_cfg=task_cfg,
    #                 model_name=MODEL_NAME_LIST[task_cfg.model_name], 
    #                 tokenizer=tokenizer,
    #                 train_mode=True if self.run_type == "train" else False,
    #                 train_data_loader=None, 
    #                 valid_data_loader=None, 
    #                 test_data_loader=None, 
    #                 model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
    #                 label_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.label),
    #                 use_cuda=self.cfg.use_cuda,
    #                 use_fp16=self.cfg.use_fp16
    #             )
                
    #             tasks.setdefault(task_name, TASK_LIST[task_name](task_config=task_config))
    #         else:
    #             task = TASK_LIST[task_name](
    #                 cfg=task_cfg,
    #                 template_dir=self.cfg.path.template
    #             )
                
    #             tasks.setdefault(task_name, task)
            
    #     dashboard = Dashboard(tasks=tasks)
    #     dashboard.run()
        
    # def cli(self):
    #     tasks = {}
        
    #     for task_name in TASK_LIST.keys():
    #         task_cfg = get_config(cfg_path=self.cfg.path.config, cfg_fn=f"{task_name}.cfg")
            
    #         if task_name in ["ner", "mrc", "sm"]:
    #             tokenizer = TOKENIZER_LIST[task_cfg.model_name].from_pretrained(MODEL_NAME_LIST[task_cfg.model_name], use_fast=True)
    #             task_config = TaskConfig(
    #                 task_name=task_name,
    #                 task_cfg=task_cfg,
    #                 model_name=MODEL_NAME_LIST[task_cfg.model_name], 
    #                 tokenizer=tokenizer,
    #                 train_mode=True if self.run_type == "train" else False,
    #                 train_data_loader=None, 
    #                 valid_data_loader=None, 
    #                 test_data_loader=None, 
    #                 model_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.model),
    #                 label_hub_path=os.path.join(self.cfg.path.root, self.cfg.path.label),
    #                 use_cuda=self.cfg.use_cuda,
    #                 use_fp16=self.cfg.use_fp16
    #             )
                
    #             tasks.setdefault(task_name, TASK_LIST[task_name](task_config=task_config))
    #         else:
    #             task = TASK_LIST[task_name](
    #                 cfg=task_cfg,
    #                 template_dir=self.cfg.path.template
    #             )
                
    #             tasks.setdefault(task_name, task)
            
    #     workflow = Workflow(tasks=tasks)
    #     workflow.cli()
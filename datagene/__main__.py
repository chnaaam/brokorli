import torch
import numpy as np
import random
import argparse

from datagene import DataGene

def fix_torch_seed(random_seed=42):
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cfg_path", type=str, default="./")
    parser.add_argument("--cfg_fn", type=str, default="datagene.cfg")
    
    parser.add_argument("--type", choices=["train", "predict"])
    
    # Train specific task
    parser.add_argument("--task_name", type=str, default="srl", choices=["srl", ])
    
    args = parser.parse_args()
    
    datagene = DataGene(
        cfg_path=args.cfg_path, 
        cfg_fn=args.cfg_fn
    )
    
    fix_torch_seed()
    
    if args.type == "train":
        datagene.train(task_name=args.task_name)
    elif args.type == "predict":
        datagene.predict()
    else:
        raise NotImplementedError()
    